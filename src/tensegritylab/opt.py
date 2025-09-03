"""Optimization utilities for prestress parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable, Tuple

import numpy as np

from .dr import dynamic_relaxation, buckling_safety_for_struts, TensegrityModel


def _scale_model_L0(model: TensegrityModel, cable_scale: float, strut_scale: float) -> TensegrityModel:
    """Return a copy of ``model`` with member rest lengths scaled.

    Parameters
    ----------
    model : TensegrityModel
        Base model whose ``L0`` values will be scaled.
    cable_scale, strut_scale : float
        Multipliers applied to cable and strut rest lengths respectively.
    """

    members = []
    for m in model.members:
        m2 = m.copy()
        if m2["kind"] == "cable":
            m2["L0"] *= cable_scale
        else:
            m2["L0"] *= strut_scale
        members.append(m2)
    return TensegrityModel(model.X, members, model.fixed)


def sweep_prestress(
    model: TensegrityModel,
    cable_scales: Iterable[float],
    strut_scales: Iterable[float],
    metric: str = "min_tension_spread",
):
    """Sweep cable/strut rest-length scales and evaluate metrics.

    Parameters
    ----------
    model : TensegrityModel
        Base structure definition.
    cable_scales, strut_scales : iterable of float
        Multipliers applied to cable and strut rest lengths.
    metric : {"min_tension_spread", "max_buckling_safety"}
        Objective metric to evaluate. ``min_tension_spread`` seeks the
        minimum spread between cable forces, while ``max_buckling_safety``
        maximizes the minimum Euler buckling safety factor of struts.

    Returns
    -------
    tuple
        ``(best_params, history)`` where ``best_params`` is a ``dict`` with
        ``cable_L0_scale``, ``strut_L0_scale`` and ``score`` keys, and
        ``history`` is a :class:`pandas.DataFrame` of all evaluated cases.
    """

    import pandas as pd

    results = []
    for cs, ss in itertools.product(cable_scales, strut_scales):
        test_model = _scale_model_L0(model, cs, ss)
        X, forces, _ = dynamic_relaxation(test_model, verbose=False)
        if metric == "min_tension_spread":
            c_forces = [abs(f["force"]) for f in forces if f["kind"] == "cable"]
            if c_forces:
                score = max(c_forces) - min(c_forces)
            else:
                score = np.inf
            better = "min"
        elif metric == "max_buckling_safety":
            buck = buckling_safety_for_struts(test_model, X, forces, EI=1.0, K=1.0)
            if buck:
                score = min(b["safety"] for b in buck)
            else:
                score = 0.0
            better = "max"
        else:
            raise ValueError("unknown metric")
        results.append({
            "cable_L0_scale": cs,
            "strut_L0_scale": ss,
            "score": score,
        })

    history = pd.DataFrame(results)
    if history.empty:
        raise ValueError("empty sweep range")

    if metric == "min_tension_spread":
        idx = history["score"].idxmin()
    else:
        idx = history["score"].idxmax()
    best_row = history.loc[idx]
    best_params = {
        "cable_L0_scale": float(best_row["cable_L0_scale"]),
        "strut_L0_scale": float(best_row["strut_L0_scale"]),
        "score": float(best_row["score"]),
    }
    return best_params, history


__all__ = ["sweep_prestress"]
