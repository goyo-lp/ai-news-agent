from __future__ import annotations

import shutil
from pathlib import Path


def _vendored_static_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "langgraphics_static"


def ensure_langgraphics_static_assets() -> Path:
    """
    Ensure official LangGraphics web assets are available in the installed
    langgraphics package at `langgraphics/static`.
    """
    source_static = _vendored_static_dir()
    source_index = source_static / "index.html"
    if not source_index.exists():
        raise RuntimeError(f"Missing vendored LangGraphics assets: {source_index}")

    import langgraphics

    package_dir = Path(langgraphics.__file__).resolve().parent
    target_static = package_dir / "static"
    target_index = target_static / "index.html"

    if target_index.exists():
        return target_static

    target_static.mkdir(parents=True, exist_ok=True)
    for child in list(target_static.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for child in source_static.iterdir():
        destination = target_static / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)

    return target_static
