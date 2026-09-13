"""Index-only contract tests; no model downloads, GPU jobs or backend processes."""
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.engines import index_tts_engine as adapter

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("index_worker_test", ROOT / "engines/index-tts/tts_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.setattr(adapter, "_check_index_tts_available", lambda _: (True, ""))
    monkeypatch.setattr(adapter, "_find_venv_python", lambda _: Path(sys.executable))
    instance = adapter.IndexTTSEngine(engine_root=str(tmp_path), temperature=0.55,
                                     emotion_strength=0.35, language="EN")
    monkeypatch.setattr(instance, "_resolve_prompt", lambda _: "reference.wav")
    return instance


def test_emotion_uses_delivery_only_and_caches():
    classifier = Mock(return_value=dict(zip(range(8), [0.1] * 8)))
    tts = SimpleNamespace(qwen_emo=SimpleNamespace(inference=classifier))
    chunk = {"delivery_instruction": "Quiet sadness.", "text": "Spoken words",
             "voice_type": "High-pitched and squeaky"}
    cache = {}
    a = worker._emotion_kwargs(tts, {}, chunk, cache)
    a["emo_vector"][0] = 99
    b = worker._emotion_kwargs(tts, {"emotion_strength": 0.2}, chunk, cache)
    classifier.assert_called_once_with("Quiet sadness.")
    assert b == {"emo_vector": [0.1] * 8, "emo_alpha": 0.2, "use_random": False}


@pytest.mark.parametrize("job,chunk", [
    ({"emotion_enabled": False}, {"delivery_instruction": "Angry."}),
    ({"emotion_strength": 0}, {"delivery_instruction": "Angry."}),
    ({}, {"delivery_instruction": ""}),
])
def test_disabled_or_absent_direction_does_not_classify(job, chunk):
    assert worker._emotion_kwargs(None, job, chunk, {}) == {}


def test_model_paths_and_worker_settings(engine):
    assert engine._model_dir.endswith("checkpoints/IndexTTS-2.5") or engine._model_dir.endswith("checkpoints\\IndexTTS-2.5")
    options = engine._worker_options()
    assert options["temperature"] == 0.55
    assert options["emotion_strength"] == 0.35
    assert options["use_bf16"] is True
    assert options["language"] == "EN"
    assert "diffusion_steps" not in options


@pytest.mark.parametrize("prebuilt", [False, True])
def test_batch_transports_direction_and_sampling(monkeypatch, engine, tmp_path, prebuilt):
    captured = {}
    callbacks = []
    class Process:
        def __init__(self, args, **kwargs):
            job = json.loads(Path(args[-1]).read_text())
            captured.update(job)
            self.stdout = io.StringIO(json.dumps({"success": True, "files": []}))
            self.stderr = io.StringIO("")
            self.returncode = 0
        def poll(self):
            return 0
        def wait(self, *args, **kwargs):
            return 0
    monkeypatch.setattr(adapter.subprocess, "Popen", Process)
    segment = {"speaker": "narrator", "chunks": ["Hello."],
               "delivery_instruction": "With restrained wonder."}
    if prebuilt:
        engine.generate_batch_prebuilt(
            worker_chunks=[{"text": "Hello.", "spk_audio_prompt": "reference.wav",
                            "output_path": str(tmp_path / "out.wav"),
                            "delivery_instruction": segment["delivery_instruction"], "_order_index": 0}],
            chunk_meta=[{"assignment": engine._voice_assignment_for({}, "narrator"), "_order_index": 0}],
            chunk_cb=lambda *args: callbacks.append(args))
    else:
        engine.generate_batch([segment], {}, tmp_path, chunk_cb=lambda *args: callbacks.append(args))
    assert captured["chunks"][0]["delivery_instruction"] == segment["delivery_instruction"]
    assert captured["temperature"] == 0.55
    assert captured["emotion_strength"] == 0.35
    assert "diffusion_steps" not in captured


def test_v25_loader_uses_correct_api_without_flash(monkeypatch):
    calls = []
    def constructor(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(device="cpu", dtype=None, qwen_emo=object())
    monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", SimpleNamespace(IndexTTS2=constructor))
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    worker._load_tts({"chunks": [{"delivery_instruction": "Calm."}]},
                    model_dir="v25", cfg_path="v25/config.yaml", model_version="IndexTTS-2.5",
                    use_fp16=True, use_accel=False, use_torch_compile=False)
    assert calls[0]["use_qwen_emo"] is True
    assert calls[0]["use_bf16"] is False
    assert "use_fp16" not in calls[0]


def test_incomplete_model_download_is_rechecked(monkeypatch, tmp_path):
    (tmp_path / "gpt.pth").touch()
    download = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    worker._ensure_model(str(tmp_path), "IndexTTS-2.5")
    assert download.call_args.kwargs["repo_id"] == "IndexTeam/IndexTTS-2.5"
    assert "*.txt" not in download.call_args.kwargs["ignore_patterns"]


def test_index_ui_has_emotion_controls_and_shared_voice_library():
    template = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    for field in ("emotion-enabled", "emotion-strength", "language", "use-bf16"):
        assert f'id="index-tts-{field}"' in template
    library = (ROOT / "static/js/library.js").read_text(encoding="utf-8")
    assert "normalizedEngine.includes('indextts')" in library


def test_install_updates_old_source_without_touching_models(monkeypatch, tmp_path):
    from scripts import install_engine as installer
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    root = tmp_path / "engines/index-tts"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("old")
    (root / "tts_worker.py").write_text("app worker")
    (root / "checkpoints").mkdir()
    (root / "checkpoints/gpt.pth").write_bytes(b"existing model")
    calls = []
    def clone(args, **kwargs):
        target = Path(args[-1])
        target.mkdir()
        (target / "pyproject.toml").write_text("new")
        (target / "uv.lock").write_text("locked")
        (target / ".git").mkdir()
    monkeypatch.setattr(installer.subprocess, "run", clone)
    monkeypatch.setattr(installer, "run", lambda args, **kwargs: calls.append(args))
    installer.install_index_tts()
    assert (root / "pyproject.toml").read_text() == "new"
    assert (root / "tts_worker.py").read_text() == "app worker"
    assert (root / "checkpoints/gpt.pth").read_bytes() == b"existing model"
    assert not (root / ".git").exists()
    assert any("--locked" in args and "3.11" in args for args in calls)
    assert any("infer_v2_5" in " ".join(args) for args in calls)
    assert (root / ".tts-story-source-revision").exists()


def test_install_failure_does_not_mark_source_ready(monkeypatch, tmp_path):
    from scripts import install_engine as installer
    monkeypatch.setattr(installer, "ROOT", tmp_path)
    root = tmp_path / "engines/index-tts"
    root.mkdir(parents=True)
    def clone(args, **kwargs):
        target = Path(args[-1])
        target.mkdir()
        (target / "pyproject.toml").touch()
    def fail(args, **kwargs):
        if "sync" in args:
            raise RuntimeError("Install failed")
    monkeypatch.setattr(installer.subprocess, "run", clone)
    monkeypatch.setattr(installer, "run", fail)
    with pytest.raises(RuntimeError):
        installer.install_index_tts()
    assert not (root / ".tts-story-source-revision").exists()
    assert not (root / ".indextts_ready").exists()


def test_index_config_round_trip_preserves_zero_and_false():
    import app
    result = app._normalize_index_tts_options({
        "index_tts_model_version": "IndexTTS-2.5",
        "index_tts_emotion_enabled": False,
        "index_tts_emotion_strength": 0,
        "index_tts_language": "ja",
        "index_tts_use_bf16": False,
    })
    assert result["index_tts_emotion_enabled"] is False
    assert result["index_tts_emotion_strength"] == 0
    assert result["index_tts_language"] == "JA"
    assert result["index_tts_use_bf16"] is False


from test_chunk_directions import review_job
from test_chunk_directions import test_direction_reaches_synthesis_and_survives_disk_reload as _verify_regen


@pytest.mark.parametrize("edit,expected", [
    (None, "Quietly, with wonder."), ("Steady and calm.", "Steady and calm."), ("", "")
])
def test_index_library_regeneration_persists_direction(monkeypatch, review_job, edit, expected):
    _verify_regen(monkeypatch, review_job, "index_tts", edit, expected)


def test_live_callback_keeps_direction_and_applies_fx_before_handoff(monkeypatch, engine, tmp_path):
    events = []
    class Process:
        def __init__(self, args, **kwargs):
            job = json.loads(Path(args[-1]).read_text())
            output = job["chunks"][0]["output_path"]
            Path(output).write_bytes(b"mock audio")
            self.stdout = io.StringIO(json.dumps({"success": True, "files": [output]}))
            self.stderr = io.StringIO(f"[CHUNK_DONE] {output}\n")
            self.returncode = 0
        def poll(self):
            return 0
        def wait(self, *args, **kwargs):
            return 0
    monkeypatch.setattr(adapter.subprocess, "Popen", Process)
    monkeypatch.setattr(engine, "_finish_chunk_audio", lambda *args: events.append("fx"))
    def callback(index, meta, path):
        assert events == ["fx"]
        assert meta["delivery_instruction"] == "Calm."
        Path(path).rename(tmp_path / "committed.wav")
        events.append("callback")
    engine.generate_batch([{"speaker": "narrator", "chunks": ["Words."],
                            "delivery_instruction": "Calm."}], {}, tmp_path, chunk_cb=callback)
    assert events == ["fx", "callback"]


def test_explicit_cuda_does_not_silently_fall_back(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was selected"):
        worker._load_tts({"device": "cuda"}, model_dir="v25", cfg_path="v25/config.yaml",
                        model_version="IndexTTS-2.5", use_fp16=False,
                        use_accel=False, use_torch_compile=False)


def test_worker_keeps_model_loaded_and_never_speaks_directions(monkeypatch, tmp_path):
    chunks = [{"text": text, "delivery_instruction": "Restrained sadness.",
               "spk_audio_prompt": "reference.wav", "output_path": str(tmp_path / f"{i}.wav")}
              for i, text in enumerate(["First sentence.", "Second sentence."])]
    job = {"chunks": chunks, "model_version": "IndexTTS-2.5", "language": "EN",
           "use_fp16": True, "use_accel": True, "emotion_strength": 0.4}
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps(job))
    classifier = Mock(return_value=dict(zip(range(8), [0.1] * 8)))
    calls = []
    def infer(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"WAV" * 100)
    tts = SimpleNamespace(qwen_emo=SimpleNamespace(inference=classifier), infer=infer)
    load = Mock(return_value=tts)
    monkeypatch.setattr(worker, "_load_tts", load)
    monkeypatch.setattr(worker, "_ensure_model", lambda *args: None)
    monkeypatch.setattr(worker, "_flash_attn_available", lambda: False)
    output = io.StringIO()
    monkeypatch.setattr(worker, "_stdout", output)
    monkeypatch.setattr(sys, "argv", ["worker", "--job-file", str(job_file)])
    worker.main()
    load.assert_called_once()
    assert load.call_args.kwargs["use_fp16"] is True
    assert load.call_args.kwargs["use_accel"] is False
    classifier.assert_called_once_with("Restrained sadness.")
    assert [c["text"] for c in calls] == ["First sentence.", "Second sentence."]
    assert all(c["emo_alpha"] == 0.4 and c["lang"] == "EN" for c in calls)
    assert all("diffusion_steps" not in c for c in calls)
    assert json.loads(output.getvalue())["success"] is True


def test_preview_uses_same_settings_and_direction(monkeypatch, engine):
    captured = {}
    def run(args, **kwargs):
        job = json.loads(Path(args[-1]).read_text())
        captured.update(job)
        output = job["chunks"][0]["output_path"]
        adapter.sf.write(output, adapter.np.zeros(200), 22050)
        return SimpleNamespace(stdout=json.dumps({"success": True, "files": [output]}))
    monkeypatch.setattr(adapter.subprocess, "run", run)
    audio = engine.generate_audio(text="Preview.", spk_audio_prompt="sample.wav",
                                  delivery_instruction="Calm.", lang_code="en")
    assert len(audio) == 200
    assert captured["temperature"] == 0.55
    assert captured["emotion_strength"] == 0.35
    assert captured["language"] == "EN"
    assert captured["chunks"][0]["delivery_instruction"] == "Calm."
    assert captured["chunks"][0]["spk_audio_prompt"] == "reference.wav"


def test_emotion_disk_cache_survives_workers_and_model_changes(tmp_path):
    model = tmp_path / "qwen0.6bemo4-merge"
    model.mkdir()
    config = model / "config.json"
    config.write_text("first")
    store = worker.EmotionStore(tmp_path)
    store.put("Quietly.", [0.2] * 8)
    second = worker.EmotionStore(tmp_path)
    assert second.get("Quietly.") == [0.2] * 8
    assert second.get("Loudly.") is None
    config.write_text("changed model")
    assert worker.EmotionStore(tmp_path).get("Quietly.") is None
    store.put("bad", [float("nan")] * 8)
    assert store.get("bad") is None


def test_cached_emotion_keeps_current_strength(tmp_path):
    store = worker.EmotionStore(tmp_path)
    store.put("Quietly.", [0.2] * 8)
    result = worker._emotion_kwargs(None, {"emotion_strength": 0.5},
                                    {"delivery_instruction": "Quietly."}, {}, store)
    assert result["emo_alpha"] == 0.5
    assert result["emo_vector"] == [0.2] * 8


def test_reference_cache_alternating_speakers_invalidation_and_limits(tmp_path):
    import torch
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    a.write_bytes(b"first")
    b.write_bytes(b"second")
    tts = SimpleNamespace(**{f: None for f in worker.ReferenceCache.fields})
    cache = worker.ReferenceCache(slots=2, max_bytes=1024)
    def capture(path, value):
        assert not cache.restore(tts, str(path))
        for field in cache.fields:
            setattr(tts, field, torch.tensor([value]))
        cache.capture(tts, str(path))
    capture(a, 1.0)
    capture(b, 2.0)
    assert cache.restore(tts, str(a))
    assert tts.cache_spk_cond.item() == 1.0
    assert tts.cache_emo_audio_prompt == str(a)
    a.write_bytes(b"new sample")
    assert not cache.restore(tts, str(a))
    assert tts.cache_spk_cond is None
    cache.slots = 1
    for field in cache.fields:
        setattr(tts, field, torch.tensor([3.0]))
    cache.capture(tts, str(a))
    assert len(cache.entries) == 1
    cache.max_bytes = 1
    cache.capture(tts, str(a))
    assert not cache.entries


def test_emotion_decode_is_bounded_inference_only_and_retries_truncation():
    import torch
    calls = []
    def generate(**kwargs):
        assert torch.is_inference_mode_enabled()
        calls.append(kwargs["max_new_tokens"])
        count = 513 if len(calls) == 1 else 5
        return torch.ones((1, count), dtype=torch.long)
    model = SimpleNamespace(device="cpu", eval=Mock(), generate=generate)
    tts = SimpleNamespace(device="cpu", qwen_emo=SimpleNamespace(
        model=model, tokenizer=SimpleNamespace(eos_token_id=2)))
    worker._configure_emotion(tts)
    result = model.generate(input_ids=torch.ones((1, 1)), max_new_tokens=32768)
    assert calls == [512, 1024]
    assert result.shape[-1] == 5
    model.eval.assert_called_once()


def test_emotion_decode_rejects_truncated_output():
    import torch
    def generate(**kwargs):
        return torch.ones((1, 1 + kwargs["max_new_tokens"]), dtype=torch.long)
    model = SimpleNamespace(device="cpu", eval=Mock(), generate=generate)
    tts = SimpleNamespace(device="cpu", qwen_emo=SimpleNamespace(
        model=model, tokenizer=SimpleNamespace(eos_token_id=2)))
    worker._configure_emotion(tts)
    with pytest.raises(RuntimeError, match="truncated"):
        model.generate(input_ids=torch.ones((1, 1)))


@pytest.mark.parametrize("strength", [0, 0.8, 1])
def test_review_strength_applies_and_persists_without_changing_job(monkeypatch, review_job, strength):
    import app
    from test_chunk_directions import write_silence_wav
    job_id, job_dir, job, chunk = review_job
    job["config_snapshot"].update(tts_engine="index_tts", index_tts_emotion_strength=0.5)
    captured = []
    class Engine:
        def generate_batch(self, output_dir, **kwargs):
            return [str(write_silence_wav(Path(output_dir) / "speech.wav", 0.1))]
    def create(engine, config):
        captured.append(dict(config))
        return Engine()
    monkeypatch.setattr(app, "get_tts_engine", create)
    app._perform_chunk_regeneration(job_id, "chunk-1", "Words.", emotion_strength=strength)
    assert captured[-1]["index_tts_emotion_strength"] == strength
    assert job["config_snapshot"]["index_tts_emotion_strength"] == 0.5
    assert app._load_chunks_metadata(job_dir)[0]["index_tts_emotion_strength"] == strength
    app._perform_chunk_regeneration(job_id, "chunk-1", "Words again.")
    assert captured[-1]["index_tts_emotion_strength"] == strength
    payload = app.app.test_client().get(f"/api/library/{job_id}/chunks").get_json()
    assert payload["chunks"][0]["index_tts_emotion_strength"] == strength


@pytest.mark.parametrize("strength", [-1, 1.1, "0.8", True, None])
def test_review_strength_rejects_invalid_values(review_job, strength):
    import app
    job_id, _, job, _ = review_job
    job["config_snapshot"]["tts_engine"] = "index_tts"
    response = app.app.test_client().post(f"/api/jobs/{job_id}/review/regen", json={
        "chunk_id": "chunk-1", "text": "Words.", "emotion_strength": strength})
    assert response.status_code == 400


def test_review_strength_is_index_only_and_queued(monkeypatch, review_job):
    import app
    job_id, _, job, _ = review_job
    calls, tasks = [], []
    monkeypatch.setattr(app.chunk_regen_executor, "submit", tasks.append)
    monkeypatch.setattr(app, "_perform_chunk_regeneration", lambda *a, **k: calls.append(k))
    client = app.app.test_client()
    data = {"chunk_id": "chunk-1", "text": "Words.", "emotion_strength": 0.8}
    assert client.post(f"/api/jobs/{job_id}/review/regen", json=data).status_code == 400
    data["engine"] = "index_tts"
    assert client.post(f"/api/jobs/{job_id}/review/regen", json=data).status_code == 200
    tasks[0]()
    assert calls[0]["emotion_strength"] == 0.8
