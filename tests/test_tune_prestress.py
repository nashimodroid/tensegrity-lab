import numpy as np

from tensegritylab.planner import plan_cables_from_struts, tune_prestress
from tensegritylab.dr import dynamic_relaxation, buckling_safety_for_struts


def _eval(model, tmin, tmax, safety):
    X, forces, _ = dynamic_relaxation(model, verbose=False)
    c_forces = [f["force"] for f in forces if f["kind"] == "cable"]
    spread = max(c_forces) - min(c_forces) if c_forces else 0.0
    pen = 0.0
    for f in forces:
        if f["kind"] == "cable":
            if f["force"] < tmin:
                pen += (tmin - f["force"]) ** 2
            if f["force"] > tmax:
                pen += (f["force"] - tmax) ** 2
    buck = buckling_safety_for_struts(model, X, forces, EI=1.0, K=1.0)
    min_s = min((b["safety"] for b in buck), default=np.inf)
    if min_s < safety:
        pen += (safety - min_s) ** 2 * 100.0
    return spread + pen, c_forces, min_s


def test_tune_prestress_improves_objective():
    B0 = np.array([0.0, 0.0, 0.0])
    B1 = np.array([1.0, 0.0, 0.0])
    B2 = np.array([0.5, np.sqrt(3) / 2, 0.0])
    T0 = B0 + np.array([0.2, 0.2, 1.0])
    T1 = B1 + np.array([0.2, 0.2, 1.0])
    T2 = B2 + np.array([0.2, 0.2, 1.0])

    struts = [
        (B0, T1),
        (B1, T2),
        (B2, T0),
    ]

    model, _ = plan_cables_from_struts(struts)

    tmin, tmax, safety = 0.5, 3.0, 1.5
    obj0, _, _ = _eval(model, tmin, tmax, safety)

    tune_prestress(model, {"cable_t": (tmin, tmax), "strut_safety": safety})

    obj1, c_forces, min_safety = _eval(model, tmin, tmax, safety)

    assert obj1 < obj0
    assert min(c_forces) >= tmin * 0.9
    assert max(c_forces) <= tmax * 1.1
    assert min_safety >= safety * 0.9
