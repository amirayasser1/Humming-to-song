from __future__ import annotations
from pathlib import Path

def list_files(folder: str, exts: tuple[str, ...]) -> list[str]:
    p = Path(folder)
    if not p.exists():
        return []
    out = []
    for ext in exts:
        out.extend([str(x) for x in p.rglob(f"*{ext}")])
    return sorted(out)

def safe_makedirs(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
