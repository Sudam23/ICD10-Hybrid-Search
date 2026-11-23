"""Guards against the container and a local install drifting apart."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirements() -> list[str]:
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _pyproject_dependencies() -> list[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["dependencies"]


def test_requirements_match_pyproject():
    assert sorted(_requirements()) == sorted(_pyproject_dependencies())


def test_every_runtime_dependency_is_pinned():
    for requirement in _requirements():
        assert "==" in requirement, f"{requirement} is not pinned to an exact version"


def test_dockerfile_installs_from_requirements():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "pip install -r requirements.txt" in dockerfile


def test_dockerfile_bakes_the_configured_model():
    from icd10_search.config import EMBEDDING_MODEL

    assert EMBEDDING_MODEL in (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_version_is_consistent():
    from icd10_search import __version__

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        assert tomllib.load(handle)["project"]["version"] == __version__


def test_gitignore_excludes_the_float32_vectors_but_keeps_the_index():
    """The 144 MB float32 file is a build by-product; the 36 MB int8 index ships."""
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert not re.search(r"^data/index/?$", ignored, re.MULTILINE)
