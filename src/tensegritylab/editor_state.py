from __future__ import annotations

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


__all__ = ["Strut", "add_strut", "edit_strut", "delete_strut"]
