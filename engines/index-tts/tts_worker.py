"""
IndexTTS worker script — runs inside the IndexTTS isolated venv.

Called by the TTS-Story engine adapter via subprocess. Reads a JSON job
from stdin (or --job-file), synthesises all chunks, writes WAV files, and
prints a JSON result to stdout.

Usage:
    python tts_worker.py --job-file /path/to/job.json

Job JSON schema:
{
    "model_dir":   "checkpoints",          # path to IndexTTS checkpoints dir
    "cfg_path":    "checkpoints/config.yaml",
    "use_fp16":    false,
    "use_deepspeed": false,
    "device":      null,                   # null = auto
    "chunks": [
        {
            "text":             "Hello world.",
            "spk_audio_prompt": "/abs/path/to/voice.wav",
            "output_path":      "/abs/path/to/chunk_0000.wav"
        },
        ...
    ]
}

Result JSON (written to stdout):
{
    "success": true,
    "files": ["/abs/path/to/chunk_0000.wav", ...]
}
or on error:
{
    "success": false,
    "error": "traceback string"
}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import traceback
import hashlib
import math
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

# Redirect stdout to stderr so that all print() calls from IndexTTS internals
# (infer_v2.py, etc.) go to stderr. The final JSON result is written directly
# to the real stdout via _stdout below.
_stdout = sys.stdout
# CLI-only redirection; importing helpers must not change process I/O.

_MODEL_REPO_MAP = {
    "IndexTTS-2.5": "IndexTeam/IndexTTS-2.5",
    "IndexTTS-2":   "IndexTeam/IndexTTS-2",
}


def _flash_attn_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _flash_attn_error(error: str) -> bool:
    lowered = error.lower()
    return "flash_attn" in lowered or "flash-attention" in lowered


def _torch_compile_available() -> bool:
    if os.environ.get("INDEXTTS_ALLOW_TORCH_COMPILE") == "1":
        return importlib.util.find_spec("triton") is not None
    if os.name == "nt":
        return False
    return importlib.util.find_spec("triton") is not None


def _torch_compile_error(error: str) -> bool:
    lowered = error.lower()
    return "tritonmissing" in lowered or "working triton installation" in lowered


def _unused_model_kwargs(error: str) -> list[str]:
    match = re.search(r"model_kwargs.*?\[(.*?)\]", error, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    return re.findall(r"['\"]([^'\"]+)['\"]", match.group(1))


def _infer_with_compatible_kwargs(tts, *, spk_audio_prompt: str, text: str, output_path: str, kwargs: dict) -> None:
    try:
        tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path=output_path,
            verbose=False,
            **kwargs,
        )
        return
    except Exception:
        error = traceback.format_exc()
        unused = [key for key in _unused_model_kwargs(error) if key in kwargs]
        if not unused or set(unused) & {"emo_vector", "emo_alpha", "use_random", "lang"}:
            raise
        for key in unused:
            kwargs.pop(key, None)
        print(
            f"[worker] IndexTTS ignored unsupported generation args {unused}; retrying without them.",
            file=sys.stderr,
            flush=True,
        )
        tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path=output_path,
            verbose=False,
            **kwargs,
        )


def _ensure_model(model_dir: str, model_version: str) -> None:
    """Download model weights from HuggingFace if not already present."""
    marker = os.path.join(model_dir, ".tts-story-download-complete")
    if os.path.isfile(marker) and os.path.isfile(os.path.join(model_dir, "config.yaml")):
        with open(marker, encoding="utf-8") as fh:
            if fh.read().strip() == model_version:
                return

    repo_id = _MODEL_REPO_MAP[model_version]
    print(
        f"[worker] Model weights not found. Downloading {repo_id} to {model_dir} ...",
        file=sys.stderr, flush=True,
    )
    print(
        f"[worker] This is a one-time download (~2-4 GB). Please wait.",
        file=sys.stderr, flush=True,
    )

    try:
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            ignore_patterns=["*.md", "examples/*"],
        )
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write(model_version)
        print(f"[worker] Model download complete.", file=sys.stderr, flush=True)
    except Exception:
        raise RuntimeError(
            f"Failed to download IndexTTS model '{repo_id}'.\n"
            f"Check your internet connection or download manually:\n"
            f"  huggingface-cli download {repo_id} --local-dir {model_dir}\n\n"
            + traceback.format_exc()
        )


def _valid_vector(vector):
    return (isinstance(vector, list) and len(vector) == 8
            and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1.2 for v in vector))


class EmotionStore:
    """Small process-safe cache; store hashed cues, not manuscript instructions."""
    def __init__(self, model_dir):
        self.db = None
        try:
            root = Path(model_dir)
            # Invalidate after replacing weights, tokenizer, config or the parser.
            paths = list((root / "qwen0.6bemo4-merge").glob("*"))
            paths += list((Path(__file__).parent / "indextts").glob("infer_v*.py"))
            signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                         for p in sorted(paths) if p.is_file()]
            self.namespace = hashlib.sha256(json.dumps(["emotion-v1", signature]).encode()).hexdigest()
            self.db = sqlite3.connect(str(root / ".tts-story-emotions.sqlite3"), timeout=2)
            self.db.execute("CREATE TABLE IF NOT EXISTS emotions (key TEXT PRIMARY KEY, vector TEXT, used REAL)")
        except (OSError, sqlite3.Error) as exc:
            print(f"[worker] Emotion disk cache unavailable: {exc}", file=sys.stderr)

    def _key(self, cue):
        return hashlib.sha256((self.namespace + cue).encode()).hexdigest()

    def get(self, cue):
        if self.db is None:
            return None
        try:
            row = self.db.execute("SELECT vector FROM emotions WHERE key=?", (self._key(cue),)).fetchone()
            vector = json.loads(row[0]) if row else None
            return vector if _valid_vector(vector) else None
        except (sqlite3.Error, ValueError, TypeError):
            return None

    def put(self, cue, vector):
        if self.db is None or not _valid_vector(vector):
            return
        try:
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO emotions VALUES (?, ?, ?)",
                                (self._key(cue), json.dumps(vector), time.time()))
                self.db.execute("DELETE FROM emotions WHERE key IN (SELECT key FROM emotions ORDER BY used DESC LIMIT -1 OFFSET 4096)")
        except sqlite3.Error as exc:
            print(f"[worker] Emotion cache write skipped: {exc}", file=sys.stderr)


def _configure_emotion(tts):
    """Retain upstream parsing/normalization; bound generation, never parse a truncation."""
    import torch
    classifier = getattr(tts, "qwen_emo", None)
    model = getattr(classifier, "model", None)
    if model is None:
        return
    model.eval()
    print(f"[worker] Emotion model device={model.device}, placement={getattr(model, 'hf_device_map', None)}", file=sys.stderr, flush=True)
    placement = getattr(model, "hf_device_map", {}) or {}
    if any(str(v) in {"cpu", "disk"} for v in placement.values()) and str(tts.device).startswith("cuda"):
        print("[worker] WARNING: emotion model is partly offloaded; emotion inference may be slow. Not forcing it onto an already occupied GPU.", file=sys.stderr, flush=True)
    original = model.generate

    def bounded_generate(*args, **kwargs):
        # Upstream requests 32768 tokens for a tiny JSON classification.
        prompt_length = kwargs["input_ids"].shape[-1]
        eos = classifier.tokenizer.eos_token_id
        for limit in (512, 1024):
            kwargs["max_new_tokens"] = limit
            started = time.perf_counter()
            with torch.inference_mode():
                result = original(*args, **kwargs)
            tokens = result.shape[-1] - prompt_length
            print(f"[worker] Emotion decode tokens={tokens} elapsed={time.perf_counter() - started:.2f}s", file=sys.stderr, flush=True)
            if tokens < limit or int(result[0, -1]) == eos:
                return result
            print("[worker] Emotion response reached token limit; retrying with a larger bounded budget.", file=sys.stderr, flush=True)
        raise RuntimeError("IndexTTS emotion classification exceeded its token budget; refusing a truncated direction result.")

    model.generate = bounded_generate


class ReferenceCache:
    """LRU of upstream conditioning tensors, bounded by count AND memory."""
    fields = ("cache_spk_cond", "cache_s2mel_style", "cache_s2mel_prompt",
              "cache_mel", "cache_emo_cond")

    def __init__(self, slots=8, max_bytes=128 * 1024 * 1024):
        self.entries = OrderedDict()
        self.slots, self.max_bytes = slots, max_bytes

    def _key(self, path):
        stat = os.stat(path)
        return (os.path.normcase(os.path.abspath(path)), stat.st_size, stat.st_mtime_ns)

    def restore(self, tts, path):
        if not all(hasattr(tts, field) for field in self.fields):
            return False
        key = self._key(path)
        entry = self.entries.pop(key, None)
        # Clear upstream's one-entry cache too, including after a file is edited.
        for field in self.fields:
            setattr(tts, field, None)
        tts.cache_spk_audio_prompt = tts.cache_emo_audio_prompt = None
        if entry is None:
            return False
        self.entries[key] = entry
        for field, value in zip(self.fields, entry[0]):
            setattr(tts, field, value)
        tts.cache_spk_audio_prompt = tts.cache_emo_audio_prompt = path
        return True

    def capture(self, tts, path):
        if not all(getattr(tts, field, None) is not None for field in self.fields):
            return
        values = tuple(getattr(tts, field).detach() for field in self.fields)
        size = sum(v.numel() * v.element_size() for v in values)
        key = self._key(path)
        self.entries.pop(key, None)
        if size > self.max_bytes:
            return
        self.entries[key] = (values, size)
        while len(self.entries) > self.slots or sum(e[1] for e in self.entries.values()) > self.max_bytes:
            self.entries.popitem(last=False)


def _emotion_kwargs(tts, job: dict, chunk: dict, cache: dict, store=None) -> dict:
    """Translate passage delivery only, never Breeze's voice-type composition."""
    cue = str(chunk.get("delivery_instruction") or "").strip()
    strength = max(0.0, min(1.0, float(job.get("emotion_strength", 0.6))))
    if not job.get("emotion_enabled", True) or not cue or strength == 0:
        return {}
    started = time.perf_counter()
    source = "memory-cache"
    if cue not in cache:
        saved = store.get(cue) if store is not None else None
        if saved is not None:
            cache[cue] = saved
            source = "disk-cache"
    if cue not in cache:
        if getattr(tts, "qwen_emo", None) is None:
            raise RuntimeError("IndexTTS emotion model is unavailable; reinstall IndexTTS or disable emotional direction.")
        import torch
        with torch.inference_mode():
            vector = list(tts.qwen_emo.inference(cue).values())
        if not _valid_vector(vector):
            raise ValueError("IndexTTS returned an invalid emotion vector")
        if len(cache) >= 128:
            cache.pop(next(iter(cache)))
        cache[cue] = vector
        source = "classified"
        if store is not None:
            store.put(cue, vector)
    while len(cache) > 128:
        cache.pop(next(iter(cache)))
    print(f"[worker] Emotion source={source} elapsed={time.perf_counter() - started:.2f}s", file=sys.stderr, flush=True)
    return {"emo_vector": list(cache[cue]), "emo_alpha": strength, "use_random": False}


def _load_tts(job, *, model_dir, cfg_path, model_version, use_fp16,
              use_accel, use_torch_compile):
    import torch
    device = job.get("device") or None
    if device == "auto":
        device = None
    if device and str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable in the IndexTTS environment. Reinstall IndexTTS with GPU support.")
    is_v25 = model_version == "IndexTTS-2.5"
    try:
        if is_v25:
            from indextts.infer_v2_5 import IndexTTS2
        else:
            from indextts.infer_v2 import IndexTTS2
    except ImportError as exc:
        raise RuntimeError(
            f"IndexTTS {model_version} runtime is incomplete. Reinstall IndexTTS in "
            "Settings > TTS Engines, then restart the backend."
        ) from exc
    options = dict(cfg_path=cfg_path, model_dir=model_dir, device=device,
                   use_deepspeed=bool(job.get("use_deepspeed", False)),
                   use_accel=use_accel, use_torch_compile=use_torch_compile,
                   use_cuda_kernel=None)
    if is_v25:
        bf16 = bool(job.get("use_bf16", True))
        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            with torch.cuda.device(device or "cuda:0"):
                bf16 = bf16 and torch.cuda.is_bf16_supported()
        elif device != "xpu":
            bf16 = False
        options["use_bf16"] = bf16
        options["use_qwen_emo"] = bool(job.get("emotion_enabled", True)) and float(job.get("emotion_strength", 0.6)) > 0 and any(
            str(c.get("delivery_instruction") or "").strip() for c in job.get("chunks", [])
        )
    else:
        options["use_fp16"] = use_fp16
    try:
        tts = IndexTTS2(**options)
    except Exception as exc:
        if _flash_attn_error(str(exc)) and options["use_accel"]:
            options["use_accel"] = False
        elif _torch_compile_error(str(exc)) and options["use_torch_compile"]:
            options["use_torch_compile"] = False
        else:
            raise
        print("[worker] Optional accelerator unavailable; retrying standard inference.", file=sys.stderr)
        tts = IndexTTS2(**options)
    print(f"[worker] {model_version} loaded: device={tts.device}, dtype={getattr(tts, 'dtype', None)}, emotion={getattr(tts, 'qwen_emo', None) is not None}", file=sys.stderr, flush=True)
    _configure_emotion(tts)
    return tts


def main() -> None:
    parser = argparse.ArgumentParser(description="IndexTTS batch worker")
    parser.add_argument("--job-file", required=True, help="Path to JSON job file")
    args = parser.parse_args()

    with open(args.job_file, "r", encoding="utf-8") as fh:
        job = json.load(fh)

    model_dir = job.get("model_dir", "checkpoints")
    model_version = job.get("model_version", "IndexTTS-2.5")
    if model_version not in {"IndexTTS-2.5", "IndexTTS-2"}:
        raise ValueError("IndexTTS worker supports versions 2 and 2.5 only")
    cfg_path = job.get("cfg_path", os.path.join(model_dir, "config.yaml"))
    use_fp16 = bool(job.get("use_fp16", False))
    use_deepspeed = bool(job.get("use_deepspeed", False))
    use_torch_compile = bool(job.get("use_torch_compile", False))
    use_accel = bool(job.get("use_accel", False))
    num_beams = int(job.get("num_beams", 3))
    diffusion_steps = int(job.get("diffusion_steps", 25))
    temperature = float(job.get("temperature", 0.8))
    top_p = float(job.get("top_p", 0.8))
    top_k = int(job.get("top_k", 30))
    repetition_penalty = float(job.get("repetition_penalty", 10.0))
    max_mel_tokens = int(job.get("max_mel_tokens", 1500))
    max_text_tokens_per_segment = int(job.get("max_text_tokens_per_segment", 120))
    device = job.get("device") or None
    chunks = job.get("chunks", [])

    if not _flash_attn_available() and use_accel:
        print(
            "[worker] flash_attn is not installed; disabling optional acceleration only.",
            file=sys.stderr,
            flush=True,
        )
        use_accel = False

    if use_torch_compile and not _torch_compile_available():
        print(
            "[worker] Triton is not available; disabling IndexTTS torch_compile "
            "for compatibility.",
            file=sys.stderr,
            flush=True,
        )
        use_torch_compile = False

    try:
        _ensure_model(model_dir, model_version)
    except Exception:
        _fail(traceback.format_exc())
        return

    try:
        tts = _load_tts(job, model_dir=model_dir, cfg_path=cfg_path,
                        model_version=model_version, use_fp16=use_fp16,
                        use_accel=use_accel, use_torch_compile=use_torch_compile)
    except Exception:
        _fail(traceback.format_exc())
        return

    emotion_cache = {}
    emotion_store = EmotionStore(model_dir)
    references = ReferenceCache()
    files: list[str] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        spk_audio_prompt = chunk.get("spk_audio_prompt", "")
        output_path = chunk.get("output_path", "")

        if not text.strip():
            print(f"[worker] Skipping empty chunk -> {output_path}", file=sys.stderr)
            continue

        try:
            generation_kwargs = {
                "num_beams": num_beams,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "max_mel_tokens": max_mel_tokens,
                "max_text_tokens_per_segment": max_text_tokens_per_segment,
            }
            generation_kwargs.update(_emotion_kwargs(tts, job, chunk, emotion_cache, emotion_store))
            if model_version == "IndexTTS-2.5":
                generation_kwargs["lang"] = job.get("language", "EN")
            reference_hit = references.restore(tts, spk_audio_prompt)
            synthesis_started = time.perf_counter()
            _infer_with_compatible_kwargs(
                tts,
                spk_audio_prompt=spk_audio_prompt,
                text=text,
                output_path=output_path,
                kwargs=generation_kwargs,
            )
            references.capture(tts, spk_audio_prompt)
            print(f"[worker] Synthesis elapsed={time.perf_counter() - synthesis_started:.2f}s reference_cache={'hit' if reference_hit else 'miss'}", file=sys.stderr, flush=True)
            if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 44:
                raise RuntimeError(f"IndexTTS produced no usable audio: {output_path}")
            files.append(output_path)
            print(f"[CHUNK_DONE] {output_path}", file=sys.stderr, flush=True)
        except Exception:
            _fail(traceback.format_exc())
            return

    result = {"success": True, "files": files}
    print(json.dumps(result), file=_stdout, flush=True)


def _fail(error: str) -> None:
    result = {"success": False, "error": error}
    print(json.dumps(result), file=_stdout, flush=True)
    sys.exit(1)


if __name__ == "__main__":
    sys.stdout = sys.stderr
    try:
        main()
    except Exception:
        _fail(traceback.format_exc())
