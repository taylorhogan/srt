"""Every Python file in the deployed tree must at least parse.

This is the 4-a.m. test. The observatory deploys by an unattended git pull at
boot; a file with a syntax error does nothing until the first import that
touches it -- which historically happens mid-night, with the roof open. Parsing
every file needs no dependencies, no hardware, and no config, so this exact
check runs identically in CI (which gates the `release` branch the observatory
pulls) and on the observatory itself.

Parse, not import: importing this codebase pulls in astropy/opencv/torch and
the live config, none of which exist on a bare CI runner, and an import-time
side effect firing in CI would be its own bug class. Syntax coverage is the
honest, zero-dependency floor. Import-with-fakes comes with the machine tests
(architecture plan, Phase 1).
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Directories that ship to the observatory. lab-style one-offs still parse
# cheaply, so they are included; only environments and runtime data are not.
EXCLUDE_PARTS = {".venv", "venv", ".git", "local", "scratch", "saved_dso",
                 "base_images", "__pycache__", "node_modules"}


def _source_files():
    for p in ROOT.rglob("*.py"):
        if EXCLUDE_PARTS.isdisjoint(p.parts):
            yield p


def test_every_python_file_parses():
    failures = []
    count = 0
    for path in _source_files():
        count += 1
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                      filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    assert count > 100, f"suspiciously few files scanned ({count}) — glob broken?"
    assert not failures, "files with syntax errors:\n" + "\n".join(failures)
