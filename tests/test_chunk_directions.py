import copy
from pathlib import Path

import pytest

import app as app_module
from src.pause_markers import write_silence_wav


@pytest.fixture
def review_job(monkeypatch, tmp_path):
    job_id = 'test-direction-review'
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    chunk = {'id': 'chunk-1', 'speaker': 'narrator', 'text': 'Original words.',
             'delivery_instruction': 'Quietly, with wonder.', 'emotion': 'Old legacy cue.',
             'relative_file': 'chunks/chunk.wav', 'engine': 'breeze_tts_2',
             'voice_assignment': {'audio_prompt_path': 'reference.wav',
                                  'extra': {'prompt_text': 'Reference transcript.',
                                            'delivery_instruction': 'Stale assignment cue.'}}}
    monkeypatch.setattr(app_module, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(app_module, '_hydrate_config_secrets', lambda config: config)
    job = {'review_mode': True, 'job_dir': str(job_dir), 'chunks': [chunk],
           'config_snapshot': {'tts_engine': 'breeze_tts_2', 'speed': 1.0, 'sample_rate': 24000},
           'voice_assignments': {}, 'word_replacements': []}
    monkeypatch.setitem(app_module.jobs, job_id, job)
    return job_id, job_dir, job, chunk


@pytest.mark.parametrize('engine', ['breeze_tts_2', 'breeze_api'])
@pytest.mark.parametrize('edit,expected', [(None, 'Quietly, with wonder.'),
                                         ('Firm and confident.', 'Firm and confident.'), ('', '')])
def test_direction_reaches_synthesis_and_survives_disk_reload(monkeypatch, review_job, engine, edit, expected):
    job_id, job_dir, job, chunk = review_job
    captured = {}

    class FakeEngine:
        def generate_batch(self, *, segments, voice_config, output_dir, **kwargs):
            captured.update(segments=copy.deepcopy(segments), voices=copy.deepcopy(voice_config))
            return [str(write_silence_wav(Path(output_dir) / 'speech.wav', 0.1))]

    monkeypatch.setattr(app_module, 'get_tts_engine', lambda *args, **kwargs: FakeEngine())
    monkeypatch.setattr(app_module.BREEZE_PRODUCTIONS, 'bind', lambda job_id, voices: voices)
    app_module._perform_chunk_regeneration(job_id, 'chunk-1', 'New words.',
                                          engine_override=engine, delivery_instruction=edit)
    assert captured['segments'][0]['delivery_instruction'] == expected
    assert captured['segments'][0]['text'] == 'New words.'
    assert captured['segments'][0]['chunks'] == ['New words.']
    assert captured['voices']['narrator']['audio_prompt_path'] == 'reference.wav'
    assert captured['voices']['narrator']['extra']['prompt_text'] == 'Reference transcript.'
    if edit is not None:
        assert 'delivery_instruction' not in captured['voices']['narrator']['extra']
    restored = app_module._load_chunks_metadata(job_dir)[0]
    assert restored['delivery_instruction'] == expected
    assert restored['engine'] == engine
    if edit == '':
        assert restored['emotion'] is None
    response = app_module.app.test_client().get(f'/api/jobs/{job_id}/chunks').get_json()
    assert response['chunks'][0]['delivery_instruction'] == expected
    library = app_module.app.test_client().get(f'/api/library/{job_id}/chunks').get_json()
    assert library['chunks'][0]['delivery_instruction'] == expected


def test_failed_generation_does_not_save_direction_edit(monkeypatch, review_job):
    job_id, job_dir, job, chunk = review_job
    before = copy.deepcopy(chunk)
    def fail(*args, **kwargs):
        raise RuntimeError('Synthetic test failure')
    monkeypatch.setattr(app_module, 'get_tts_engine', fail)
    with pytest.raises(RuntimeError, match='Synthetic test failure'):
        app_module._perform_chunk_regeneration(job_id, 'chunk-1', 'New words.', delivery_instruction='Changed cue')
    assert chunk == before
    assert not (job_dir / 'chunks_metadata.json').exists()


@pytest.mark.parametrize('edit', [None, '', 'Steady, with restrained tension.'])
def test_regen_api_passes_direction_through_executor(monkeypatch, review_job, edit):
    job_id, _, _, _ = review_job
    submitted = []
    calls = []
    monkeypatch.setattr(app_module.chunk_regen_executor, 'submit', submitted.append)
    monkeypatch.setattr(app_module, '_update_regen_status', lambda *a, **k: None)
    monkeypatch.setattr(app_module, '_perform_chunk_regeneration', lambda *a, **k: calls.append(k))
    payload = {'chunk_id': 'chunk-1', 'text': 'Original words.'}
    if edit is not None:
        payload['delivery_instruction'] = edit
    response = app_module.app.test_client().post(f'/api/jobs/{job_id}/review/regen', json=payload)
    assert response.status_code == 200
    submitted[0]()
    assert calls[0]['delivery_instruction'] == edit


@pytest.mark.parametrize('edit', [None, 123, {}, '[direction]Softly[/direction]'])
def test_invalid_direction_is_rejected(review_job, edit):
    job_id, _, _, _ = review_job
    response = app_module.app.test_client().post(f'/api/jobs/{job_id}/review/regen', json={
        'chunk_id': 'chunk-1', 'text': 'Original words.', 'delivery_instruction': edit})
    assert response.status_code == 400


def test_legacy_cue_and_explicit_clear_serialization():
    serialize = app_module._serialize_chunk_for_response
    assert serialize('test', {'emotion': 'Softly'})['delivery_instruction'] == 'Softly'
    assert serialize('test', {'delivery_instruction': None, 'emotion': 'Softly'})['delivery_instruction'] == 'Softly'
    assert serialize('test', {'delivery_instruction': '', 'emotion': 'Softly'})['delivery_instruction'] == ''
    assert serialize('test', {})['delivery_instruction'] == ''
