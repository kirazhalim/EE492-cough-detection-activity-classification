from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward from a start path."""
    current = Path.cwd() if start is None else Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "data").exists():
            return candidate

    raise FileNotFoundError("Could not find project root containing .git and data/.")


def project_path(*parts: str, root: str | Path | None = None) -> Path:
    """Build an absolute path inside the project root."""
    return find_project_root(root) / Path(*parts)

