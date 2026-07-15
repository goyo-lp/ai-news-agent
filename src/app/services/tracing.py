from __future__ import annotations

from typing import Any, Callable


try:
    from langsmith import traceable as _traceable
except Exception:  # pragma: no cover
    def _traceable(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:  # type: ignore[no-redef]
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


def traceable(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """LangSmith's @traceable decorator, or a no-op passthrough if langsmith isn't
    installed — lets every node use the same decorator regardless of whether
    observability is configured."""
    return _traceable(*args, **kwargs)  # type: ignore[no-any-return]
