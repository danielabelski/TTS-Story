import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import app as app_module
from src.directed_prompt_presets import PROMPT_DIR, with_directed_presets
from src.voice_directions import attach_speaker_profiles, compose_voice_direction
from src.engines.base import VoiceAssignment
from src.engines.breeze_tts_2_engine import BreezeTTS2Engine
from src.engines.breeze_api_engine import BreezeAPIEngine
from src.pause_markers import write_silence_wav

PROFILE = {'description': 'A boastful woodland creature.',
           'voice': 'High-pitched, squeaky, comically pompous.',
           'voice_design_prompt': 'MALE VOICE. Light, squeaky timbre.'}


@pytest.mark.parametrize('cue', ['Spoken with self-important confidence.',
                                 'Delivered with theatrical mystery and importance.',
                                 'Exclaimed with astonished delight, then grand approval.', ''])
def test_identity_is_prepended_once_without_touching_cue(cue):
    extra = {'speaker_profile': PROFILE}
    combined = compose_voice_direction(extra, cue)
    assert combined == f"{PROFILE['voice']} {cue}".strip()
    assert compose_voice_direction(extra, combined) == combined
    assert PROFILE['description'] not in combined
    assert 'MALE' not in combined


def test_legacy_profile_and_empty_type_keep_direction_unchanged():
    assert compose_voice_direction({}, 'Softly.') == 'Softly.'
    assert compose_voice_direction({'speaker_profile': {'voice': ''}}, 'Softly.') == 'Softly.'
    assert compose_voice_direction({'speaker_profile': {'voice': 'Squeaky'}}, 'Brightly.') == 'Squeaky. Brightly.'


def test_profiles_snapshot_normalized_speaker_ids_without_secrets_or_alias_changes():
    original = {'barnaby-male': {'audio_prompt_path': 'b.wav', 'extra': {'prompt_text': 'Hello.'}},
                'narrator': {'voice': 'other'}}
    result = attach_speaker_profiles(original, {'Barnaby-Male': {**PROFILE, 'api_key': 'excluded'}})
    assert result['barnaby-male']['extra']['speaker_profile'] == PROFILE
    assert result['barnaby-male']['extra']['prompt_text'] == 'Hello.'
    assert 'extra' not in result['narrator']
    assert 'speaker_profile' not in original['barnaby-male']['extra']


def test_v2_backup_preserved_and_v3_added_without_overwriting_customizations():
    v2 = (PROMPT_DIR / 'strict-book-conversion-v2-directed.txt').read_text(encoding='utf-8').strip()
    backup = (PROMPT_DIR / 'backups/strict-book-conversion-v2-directed-20260911.txt').read_text(encoding='utf-8').strip()
    assert backup == v2
    original = [{'id': 'strict-book-conversion-v2-directed', 'title': 'My V2', 'prompt': 'Custom unchanged'}]
    presets = with_directed_presets(original)
    assert presets[0] == original[0]
    assert len(original) == 1
    assert with_directed_presets(presets) == presets
    v3 = next(p['prompt'] for p in presets if p['id'] == 'strict-book-conversion-v3-directed')
    assert 'STABLE VOICE TYPE VERSUS PASSAGE DELIVERY' in v3
    assert 'Do NOT prepend Voice Type yourself' in v3
    assert 'Every [narrator] has its own [/narrator]' in v3


def test_profile_prompt_requires_stable_sound_separate_from_delivery():
    prompt = app_module.compose_gemini_speaker_profile_prompt('Custom prompt', ['barnaby-male'])
    assert 'stable audible identity' in prompt
    assert 'Do not include gender labels' in prompt
    assert 'same stable Voice Type' in prompt


@pytest.mark.parametrize('reference', [False, True])
def test_local_q8_receives_combined_instruction_and_unchanged_seed(monkeypatch, reference):
    engine = BreezeTTS2Engine.__new__(BreezeTTS2Engine)
    engine.default_prompt = None
    engine.default_prompt_text = ''
    engine.default_instruction = 'Default design'
    engine._sample_rate = 24000
    engine.design_cfg_scale = 3
    engine.direction_cfg_scale = 4
    engine.clone_cfg_scale = 2
    engine._resolve_prompt = lambda value: value
    monkeypatch.setattr('src.engines.breeze_tts_2_engine.convert_mp3_to_wav_if_needed', lambda value: (value, None))
    engine.runtime_kind = 'q8'
    engine.max_new_tokens = 100
    engine.runtime = SimpleNamespace(generate=Mock(return_value=np.zeros(240)))
    engine.post_processor = SimpleNamespace(apply_post_pipeline=lambda audio, *args: audio)
    assignment = VoiceAssignment(audio_prompt_path='sample.wav' if reference else None,
        extra={'speaker_profile': PROFILE, 'delivery_instruction': 'Spoken confidently.', 'prompt_text': 'Reference transcript.'})
    engine._synthesize('Hello.', assignment, 1234)
    args = engine.runtime.generate.call_args.args
    assert args[0] == 'Hello.'
    assert args[1] == PROFILE['voice'] + ' Spoken confidently.'
    assert args[5] == 1234
    assert args[2] == ('sample.wav' if reference else None)
    assert args[4] == (4 if reference else 3)
    assert assignment.extra['delivery_instruction'] == 'Spoken confidently.'


def test_hosted_breeze_receives_combined_instruction_without_mutating_metadata(monkeypatch):
    engine = BreezeAPIEngine('test-key')
    engine.request = Mock(return_value=SimpleNamespace(content=b'audio'))
    engine._audio_converter = lambda *a, **k: b'wav'
    monkeypatch.setattr('src.engines.breeze_api_engine.apply_wav_effects', lambda data, fx: data)
    items = engine._work_items([{'speaker': 'b', 'chunks': ['Hello.'], 'delivery_instruction': 'Softly.'}],
                              {'b': {'voice': 'voc_b', 'extra': {'speaker_profile': PROFILE}}})
    engine._synthesize(items[0]['text'], items[0]['assignment'])
    assert engine.request.call_args.kwargs['json']['instructions'] == PROFILE['voice'] + ' Softly.'
    assert items[0]['delivery_instruction'] == 'Softly.'


@pytest.fixture
def production(monkeypatch, tmp_path):
    job_id = 'test-voice-identity'
    monkeypatch.setattr(app_module, 'OUTPUT_DIR', tmp_path)
    tmp_path = tmp_path / job_id
    tmp_path.mkdir()
    chunk = {'id': 'c1', 'speaker': 'barnaby-male', 'text': 'Hello.', 'relative_file': 'chunks/c1.wav',
             'delivery_instruction': 'Softly.', 'voice_assignment': {'audio_prompt_path': 'b.wav',
                'extra': {'prompt_text': 'Sample transcript.'}}}
    other = {**copy.deepcopy(chunk), 'id': 'c2', 'speaker': 'narrator'}
    job = {'status': 'completed', 'review_mode': True, 'job_dir': str(tmp_path), 'chunks': [chunk, other],
           'voice_assignments': {}, 'job_payload': {}, 'config_snapshot': {'tts_engine': 'breeze_tts_2'}}
    monkeypatch.setitem(app_module.jobs, job_id, job)
    monkeypatch.setattr(app_module, '_persist_job_state', lambda *a, **k: None)
    monkeypatch.setattr(app_module, '_hydrate_config_secrets', lambda config: config)
    return job_id, job, tmp_path


def test_save_properties_persist_and_regenerate_with_replacement_voice(monkeypatch, production):
    job_id, job, path = production
    response = app_module.app.test_client().post(f'/api/jobs/{job_id}/review/speaker-profile',
                json={'speaker': 'barnaby-male', 'profile': PROFILE})
    assert response.status_code == 200
    restored = app_module._load_chunks_metadata(path)
    assert restored[0]['voice_assignment']['extra']['speaker_profile'] == PROFILE
    assert restored[0]['delivery_instruction'] == 'Softly.'
    assert 'speaker_profile' not in restored[1]['voice_assignment']['extra']
    assert job['job_payload']['voice_assignments']['barnaby-male']['extra']['speaker_profile'] == PROFILE
    job['chunks'] = restored
    job['voice_assignments'] = {}  # Simulate Library restoring only chunk metadata.
    captured = {}
    class Engine:
        def generate_batch(self, **kwargs):
            captured.update(kwargs)
            return [str(write_silence_wav(Path(kwargs['output_dir']) / 'new.wav', .1))]
    monkeypatch.setattr(app_module, 'get_tts_engine', lambda *a, **k: Engine())
    app_module._perform_chunk_regeneration(job_id, 'c1', 'Updated.',
        voice_override={'audio_prompt_path': 'new.wav', 'extra': {'prompt_text': 'New transcript.'}},
        delivery_instruction='Confidently.')
    extra = captured['voice_config']['barnaby-male']['extra']
    assert extra['speaker_profile'] == PROFILE
    assert extra['prompt_text'] == 'New transcript.'
    assert captured['segments'][0]['delivery_instruction'] == 'Confidently.'
    assert app_module._load_chunks_metadata(path)[0]['voice_assignment']['extra']['speaker_profile'] == PROFILE


def test_save_blocks_busy_jobs_and_rolls_back_disk_failure(monkeypatch, production):
    job_id, job, _ = production
    client = app_module.app.test_client()
    payload = {'speaker': 'barnaby-male', 'profile': PROFILE}
    job['regen_tasks'] = {'c1': {'status': 'running'}}
    assert client.post(f'/api/jobs/{job_id}/review/speaker-profile', json=payload).status_code == 409
    job['regen_tasks'] = {}
    before = copy.deepcopy(job)
    monkeypatch.setattr(app_module, 'write_json_atomic', Mock(side_effect=OSError('Disk full')))
    assert client.post(f'/api/jobs/{job_id}/review/speaker-profile', json=payload).status_code == 500
    assert job == before


def test_clearing_voice_type_is_persisted(production):
    job_id, job, path = production
    client = app_module.app.test_client()
    endpoint = f'/api/jobs/{job_id}/review/speaker-profile'
    client.post(endpoint, json={'speaker': 'barnaby-male', 'profile': PROFILE})
    assert client.post(endpoint, json={'speaker': 'barnaby-male', 'profile': {}}).status_code == 200
    assert app_module._load_chunks_metadata(path)[0]['voice_assignment']['extra']['speaker_profile']['voice'] == ''


def test_main_payload_and_library_editor_expose_profiles():
    root = Path(__file__).resolve().parents[1]
    main = (root / 'static/js/main.js').read_text(encoding='utf-8')
    library = (root / 'static/js/library.js').read_text(encoding='utf-8')
    assert 'speaker_profiles: Object.fromEntries(Object.keys(voiceAssignments)' in main
    assert '[speaker, findSpeakerProfile(speaker).profile]' in main
    for text in ['speaker-profile-description', 'speaker-profile-voice', 'speaker-profile-design',
                 '/review/speaker-profile', 'Save Speaker Properties']:
        assert text in library
