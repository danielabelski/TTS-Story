"""Add versioned built-in prompts without overwriting users' preset edits."""
import copy
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "docs" / "prompts"


def with_directed_presets(presets):
    result = copy.deepcopy(presets) if isinstance(presets, list) else []
    ids = {p.get("id") for p in result if isinstance(p, dict)}
    for preset_id, title, filename in (
        ("strict-book-conversion-v2-directed", "Strict Book Conversion V2 - Directed",
         "strict-book-conversion-v2-directed.txt"),
        ("strict-book-conversion-v2-directed-backup-20260911",
         "Strict Book Conversion V2 - Directed (Backup 2026-09-11)",
         "backups/strict-book-conversion-v2-directed-20260911.txt"),
        ("strict-book-conversion-v3-directed", "Strict Book Conversion V3 - Directed",
         "strict-book-conversion-v3-directed.txt"),
    ):
        if preset_id not in ids:
            result.append({"id": preset_id, "title": title,
                           "prompt": (PROMPT_DIR / filename).read_text(encoding="utf-8").strip()})
    return result
