"""OmniVoice isolated worker — called via subprocess from the main app.

Accepts a JSON job file via --job-file.  Writes output WAV files and prints
progress markers to stderr so the parent process can track completion.

Job file schema
---------------
{
  "mode": "clone" | "design",

  // clone mode:
  "chunks": [
    {
      "text": "...",
      "ref_audio": "/path/to/prompt.wav",
      "ref_text": "optional transcript",
      "output_path": "/path/to/chunk_0000.wav",
      "_order_index": 0
    }
  ],

  // design mode (single preview):
  "text": "...",
  "instruct": "female, low pitch, british accent",
  "output_path": "/path/to/preview.wav",

  // shared:
  "model_id": "k2-fsa/OmniVoice",
  "device": "auto",   // auto | cuda | cpu
  "dtype": "float16", // float16 | bfloat16 | float32
  "num_step": 32,
  "speed": 1.0,
  "post_process": true   // set to false to disable silence-trimming post-processing
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice  # type: ignore


def _resolve_device(device: str) -> str:
    d = (device or "auto").strip().lower()
    if d == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return d


def _resolve_dtype(dtype: str) -> torch.dtype:
    d = (dtype or "float16").strip().lower()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _ensure_model(model_id: str) -> str:
    """Download model if not cached locally; return local path string."""
    local_dir = Path(__file__).resolve().parent.parent.parent / "models" / "omnivoice"
    local_dir.mkdir(parents=True, exist_ok=True)
    model_path = local_dir / model_id.replace("/", "_")
    if not model_path.exists() or not any(model_path.iterdir()):
        print(f"[omnivoice_worker] Downloading model {model_id} ...", file=sys.stderr)
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
    return str(model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args()

    with open(args.job_file, "r", encoding="utf-8") as f:
        job = json.load(f)

    model_id = job.get("model_id") or "k2-fsa/OmniVoice"
    device = _resolve_device(job.get("device") or "auto")
    dtype = _resolve_dtype(job.get("dtype") or "float16")
    num_step = int(job.get("num_step") or 32)
    speed = float(job.get("speed") or 1.0)
    post_process = job.get("post_process", True)
    mode = job.get("mode") or "clone"

    model_path = _ensure_model(model_id)
    print(f"[omnivoice_worker] Loading model from {model_path} (device={device})", file=sys.stderr)

    model = OmniVoice.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
    )

    sample_rate = 24000

    if mode == "clone":
        chunks = job.get("chunks") or []
        for chunk in chunks:
            text = chunk["text"]
            ref_audio = chunk["ref_audio"]
            ref_text = chunk.get("ref_text") or None
            output_path = chunk["output_path"]

            kwargs = dict(text=text, ref_audio=ref_audio, num_step=num_step, speed=speed)
            if ref_text:
                kwargs["ref_text"] = ref_text
            if not post_process:
                kwargs["post_process"] = False

            audio_list = model.generate(**kwargs)
            audio = np.asarray(audio_list[0], dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, sample_rate)
            print(f"[CHUNK_DONE] {output_path}", file=sys.stderr)

    elif mode == "design":
        text = job["text"]
        instruct = job["instruct"]
        output_path = job["output_path"]

        design_kwargs = dict(text=text, instruct=instruct, num_step=num_step, speed=speed)
        if not post_process:
            design_kwargs["post_process"] = False
        audio_list = model.generate(**design_kwargs)
        audio = np.asarray(audio_list[0], dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sample_rate)
        # Print the output path as the result
        print(output_path)
        print(f"[DESIGN_DONE] {output_path}", file=sys.stderr)

    else:
        print(f"[omnivoice_worker] Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
