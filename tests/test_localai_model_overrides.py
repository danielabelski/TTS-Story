import ast
from pathlib import Path


def test_model_override_preserves_case_and_ignores_connection_settings():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_normalize_engine_options')
    namespace = {'Dict': dict, 'Any': object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'app.py', 'exec'), namespace)
    normalize = namespace['_normalize_engine_options']
    assert normalize('localai_tts', {'localai_tts_model': ' Qwen/Custom ', 'localai_tts_api_key': 'ignored'}) == {'localai_tts_model': 'Qwen/Custom'}
    assert normalize('localai_tts', {'localai_tts_model': ''}) == {}
    assert normalize('localai_tts', {'localai_tts_model': 123}) == {}
