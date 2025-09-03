from __future__ import annotations

import json
from typing import List, Sequence, Tuple

Strut = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


def add_strut(struts: Sequence[Strut], a: Sequence[float], b: Sequence[float]) -> List[Strut]:
    """Return a new list with a strut appended."""
    new = list(struts)
    new.append((tuple(a), tuple(b)))
    return new


def edit_strut(struts: Sequence[Strut], index: int, a: Sequence[float], b: Sequence[float]) -> List[Strut]:
    """Return a new list with the strut at *index* replaced."""
    new = list(struts)
    new[index] = (tuple(a), tuple(b))
    return new


def delete_strut(struts: Sequence[Strut], index: int) -> List[Strut]:
    """Return a new list with the strut at *index* removed."""
    new = list(struts)
    del new[index]
    return new


def struts_to_json(struts: Sequence[Strut]) -> str:
    """Serialize *struts* to a JSON string."""
    return json.dumps(struts)


def struts_from_json(data: str) -> List[Strut]:
    """Deserialize *data* into a list of struts."""
    raw = json.loads(data)
    return [
        ((float(a[0]), float(a[1]), float(a[2])), (float(b[0]), float(b[1]), float(b[2])))
        for a, b in raw
    ]


__all__ = [
    "Strut",
    "add_strut",
    "edit_strut",
    "delete_strut",
    "struts_to_json",
    "struts_from_json",
]
