import numpy as np

from tensegritylab.planner import plan_cables_from_struts
from tensegritylab.dr import dynamic_relaxation, degree_check


def test_planner_basic():
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

    model = plan_cables_from_struts(struts)

    cables = [m for m in model.members if m.kind == "cable"]
    assert len(cables) > 0
    assert len(model.fixed) >= 3

    ok, _ = degree_check(model)
    assert ok

    _, _, info = dynamic_relaxation(model, tol=1e-4, max_steps=20000, verbose=False)
    assert info["rms"] < 1e-4
