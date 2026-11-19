"""Test that version is consistent across pyproject.toml, __init__.py, and CHANGELOG.md."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _pyproject_version() -> str:
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _init_version() -> str:
    from sherox import __version__

    return __version__


def _changelog_version() -> str:
    # First released heading ("## [x.y.z]"); "## [Unreleased]" may precede it.
    text = (ROOT / "CHANGELOG.md").read_text()
    m = re.search(r"^## \[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)
    assert m, "Could not find a released version heading in CHANGELOG.md"
    return m.group(1)


def test_version_consistency():
    pyproject = _pyproject_version()
    init = _init_version()
    changelog = _changelog_version()

    assert pyproject == init, f"pyproject.toml ({pyproject}) != __init__.py ({init})"
    assert (
        pyproject == changelog
    ), f"pyproject.toml ({pyproject}) != CHANGELOG.md ({changelog})"
