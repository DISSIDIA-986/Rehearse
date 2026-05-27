"""CLI dispatch for markdown-recall mode — the validation branch (no mic/models)."""

from localvocal.main_loop import main


def test_markdown_requires_path(capsys):
    assert main(["--content", "markdown"]) == 1
    assert "--path" in capsys.readouterr().err


def test_markdown_rejects_missing_file(capsys):
    assert main(["--content", "markdown", "--path", "/no/such/file.md"]) == 1
    assert "No such file" in capsys.readouterr().err
