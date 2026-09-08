"""AI News Agent runtime foundation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-news-agent")
except PackageNotFoundError:  # pragma: no cover - editable installs provide it
    __version__ = "0.0.0"

__all__ = ["__version__"]
