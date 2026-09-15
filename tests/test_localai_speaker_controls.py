"""Guard against hiding LocalAI's freeform fields inside another engine's panel."""
from html.parser import HTMLParser
from pathlib import Path


class Panels(HTMLParser):
    def __init__(self):
        super().__init__()
        self.divs = []
        self.ancestors = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            role = attrs.get("data-role")
            if role:
                self.ancestors[role] = list(self.divs)
            self.divs.append(role)
        if tag == "input" and "localai-" in attrs.get("class", ""):
            self.ancestors[attrs["class"]] = list(self.divs)

    def handle_endtag(self, tag):
        if tag == "div" and self.divs:
            self.divs.pop()


def test_localai_fields_are_independent_of_qwen_panel():
    source = (Path(__file__).resolve().parents[1] / "static/js/main.js").read_text(encoding="utf-8")
    start = source.index('<div class="qwen3-inline-options"')
    end = source.index('<div class="azure-speech-inline-options"', start)
    panels = Panels()
    panels.feed(source[start:end])
    assert "qwen3-control" not in panels.ancestors["localai-control"]
    for field in ("localai-voice-input", "localai-language-input"):
        assert "localai-control" in panels.ancestors[field]
        assert "qwen3-control" not in panels.ancestors[field]


def test_localai_sample_picker_visible_and_synchronized():
    source = (Path(__file__).resolve().parents[1] / "static/js/main.js").read_text(encoding="utf-8")
    assert "const showCatalogControl = isLocalAITts ||" in source
    assert 'class="localai-sample-filter"' in source
    assert 'class="help-text localai-sample-info"' in source
    assert "if (input) input.value = selectedVoice;" in source
    assert "requestedModel !== currentModel" in source


def test_manual_voice_field_does_not_autofill_default():
    source = (Path(__file__).resolve().parents[1] / "static/js/main.js").read_text(encoding="utf-8")
    assert 'placeholder="Type a voice name or ID here (e.g. Ryan)"' in source
    assert "input.value = currentValue || row.querySelector('.voice-select')" not in source
    assert "localAIVoice.value = assignments[speaker]?.voice || '';" in source
