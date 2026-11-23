"""Download the public CMS ICD-10-CM 2026 release.

The files are published by the U.S. Centers for Medicare & Medicaid Services
and are in the public domain (works of the U.S. federal government). Only the
three members the pipeline actually needs are extracted; the rest of each
archive (PDF renderings, drug/neoplasm tables) is left untouched.

A manifest with SHA-256 digests is written next to the downloads so that a
re-run can prove it is working on the same bytes as a previous run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from icd10_search.config import (
    CMS_DOWNLOADS,
    INDEX_XML_MEMBER,
    ORDER_FILE_MEMBER,
    RAW_DIR,
    TABULAR_XML_MEMBER,
)

USER_AGENT = "icd10-hybrid-search/1.0 (open-source demo project)"
WANTED_MEMBERS = {ORDER_FILE_MEMBER, TABULAR_XML_MEMBER, INDEX_XML_MEMBER}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (fixed https URLs)
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _extract_wanted(archive: Path, target_dir: Path) -> list[Path]:
    """Extract only the known member names, flattened, ignoring archive paths.

    Flattening by basename also defends against path traversal in the archive.
    """
    extracted: list[Path] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name not in WANTED_MEMBERS:
                continue
            out = target_dir / name
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out)
    return extracted


def ensure_raw_data(raw_dir: Path = RAW_DIR, *, force: bool = False) -> dict[str, Path]:
    """Download and extract the CMS source files; return a map of member -> path."""
    extract_dir = raw_dir / "extracted"
    manifest_path = raw_dir / "download_manifest.json"
    manifest: dict[str, dict[str, str]] = {}

    for filename, url in CMS_DOWNLOADS.items():
        archive = raw_dir / filename
        if force or not archive.exists():
            print(f"[download] {url}")
            _download(url, archive)
        else:
            print(f"[download] cached {archive.name}")
        manifest[filename] = {
            "url": url,
            "sha256": _sha256(archive),
            "bytes": str(archive.stat().st_size),
        }
        _extract_wanted(archive, extract_dir)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    resolved = {name: extract_dir / name for name in WANTED_MEMBERS}
    missing = [name for name, path in resolved.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"expected members missing from CMS archives: {missing}")
    return resolved


if __name__ == "__main__":  # pragma: no cover - manual entry point
    for member, path in ensure_raw_data().items():
        print(f"{member}: {path} ({path.stat().st_size:,} bytes)")
