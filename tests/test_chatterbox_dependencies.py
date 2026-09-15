"""Dependency diagnostics without importing Torch or downloading TTS models."""
import argparse
import ast
import sys
import traceback
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "engines/chatterbox/chatterbox_worker.py"


def worker_functions():
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"validate_watermarker", "main"}]
    namespace = dict(argparse=argparse, sys=sys, traceback=traceback)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(WORKER), "exec"), namespace)
    return namespace


def test_missing_dependency_explains_repair(monkeypatch):
    monkeypatch.setitem(sys.modules, "perth", None)
    with pytest.raises(RuntimeError, match="Settings > TTS Engines") as error:
        worker_functions()["validate_watermarker"]()
    assert isinstance(error.value.__cause__, ImportError)
    assert "setuptools<81" in str(error.value)


@pytest.mark.parametrize("available", [True, False])
def test_perth_must_be_callable(monkeypatch, available):
    factory = Mock()
    perth = ModuleType("perth")
    perth.PerthImplicitWatermarker = factory if available else None
    module = ModuleType("perth.perth_net.perth_net_implicit.perth_watermarker")
    module.PerthImplicitWatermarker = factory
    monkeypatch.setitem(sys.modules, "perth", perth)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if available:
        assert worker_functions()["validate_watermarker"]() is factory
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            worker_functions()["validate_watermarker"]()


def test_environment_check_constructs_on_cpu(monkeypatch, capsys):
    namespace = worker_functions()
    factory = Mock()
    namespace.update(validate_watermarker=lambda: factory, torch=Mock(__version__="test"))
    monkeypatch.setattr(sys, "argv", ["worker", "--check-env"])
    assert namespace["main"]() == 0
    factory.assert_called_once_with(device="cpu")
    assert "Perth watermarking verified" in capsys.readouterr().out


def test_environment_failure_reports_traceback(monkeypatch, capsys):
    namespace = worker_functions()
    namespace["validate_watermarker"] = Mock(side_effect=RuntimeError("broken dependency"))
    monkeypatch.setattr(sys, "argv", ["worker", "--check-env"])
    assert namespace["main"]() == 1
    output = capsys.readouterr().err
    assert "CHATTERBOX_WORKER_ERROR" in output
    assert "Traceback (most recent call last)" in output
    assert "broken dependency" in output


def test_chatterbox_pins_legacy_resource_dependency():
    requirements = (ROOT / "requirements-engines/chatterbox_turbo_local.txt").read_text()
    assert "setuptools>=65,<81" in requirements.splitlines()
