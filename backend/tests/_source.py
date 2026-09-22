"""Source of a method plus every ``self.<method>`` it reaches.

Wiring tests assert on the text of ``predict`` / ``_train_horizon_inline``.
Those bodies were split into private helpers (21eaf4e), so reading one
method's source misses the code it delegates to. Following ``self.`` calls
keeps the assertions about the pipeline, not about where a line happens to
sit today.
"""

from __future__ import annotations

import inspect
import re

_SELF_CALL = re.compile(r"self\.(\w+)\(")


def method_closure_source(cls: type, name: str) -> str:
    seen: list[str] = []
    queue = [name]
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        member = inspect.getattr_static(cls, current, None)
        func = getattr(member, "__func__", member)
        if not inspect.isfunction(func):
            continue
        seen.append(current)
        queue.extend(_SELF_CALL.findall(inspect.getsource(func)))
    return "\n".join(inspect.getsource(getattr(cls, n)) for n in seen)
