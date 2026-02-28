from __future__ import annotations

from typing import Any, Callable


try:
    from langsmith import traceable as _traceable
except Exception:  # pragma: no cover
    def _traceable(*_args: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


def traceable(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _traceable(*args, **kwargs)
