"""Launcher-menu tests: pure argv mapping + the no-launch paths (no mic/models)."""

import builtins

import localvocal.main_loop as ml
from localvocal.main_loop import build_menu_argv, run_menu


def test_build_menu_argv_mapping():
    assert build_menu_argv("1") == []
    assert build_menu_argv("2") == ["--manual-turns"]
    assert build_menu_argv("3") == ["--manual-turns", "--brief"]
    assert build_menu_argv("4", "/x.md") == ["--content", "markdown", "--path", "/x.md"]
    assert build_menu_argv("5") == ["--smoke"]
    assert build_menu_argv("6") is None      # quit handled by run_menu
    assert build_menu_argv("9") is None      # invalid


def _feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda *a: next(it))


def test_run_menu_quit(monkeypatch, capsys):
    _feed(monkeypatch, ["6"])
    assert run_menu() == 0
    assert "pick a mode" in capsys.readouterr().out


def test_run_menu_eof_quits(monkeypatch):
    def boom(*a):
        raise EOFError
    monkeypatch.setattr(builtins, "input", boom)
    assert run_menu() == 0


def test_run_menu_reprompts_then_quits(monkeypatch, capsys):
    _feed(monkeypatch, ["99", "6"])  # invalid -> reprompt -> quit
    assert run_menu() == 0
    assert "1 to 6" in capsys.readouterr().out


def test_run_menu_markdown_bad_path_reprompts(monkeypatch, capsys):
    # choose markdown, give a missing file, then quit
    _feed(monkeypatch, ["4", "/no/such/file.md", "6"])
    assert run_menu() == 0
    assert "No such file" in capsys.readouterr().out


def test_run_menu_dispatches(monkeypatch):
    # choosing 5 should call main(["--smoke"]) — stub main to capture, no models
    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(ml, "main", fake_main)
    _feed(monkeypatch, ["5"])
    assert run_menu() == 0
    assert captured["argv"] == ["--smoke"]
