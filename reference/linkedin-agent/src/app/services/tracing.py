from __future__ import annotations

from typing import Any, Callable, TypeAlias, cast

TraceableFactory: TypeAlias = Callable[..., Callable[[Callable[..., Any]], Callable[..., Any]]]
_traceable_impl: TraceableFactory | None

try:
    from langsmith import traceable as _imported_traceable
except Exception:  # pragma: no cover
    _traceable_impl = None
else:
    _traceable_impl = cast(TraceableFactory, _imported_traceable)


def traceable(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if _traceable_impl is None:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator
    return cast(TraceableFactory, _traceable_impl)(*args, **kwargs)
