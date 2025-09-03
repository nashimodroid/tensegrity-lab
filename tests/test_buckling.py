import numpy as np
from tensegritylab.dr import (
    build_snelson_prism,
    dynamic_relaxation,
    buckling_safety_for_struts,
)


def test_buckling_safety_positive_and_skip_zero():
    model = build_snelson_prism()
    X, forces, _ = dynamic_relaxation(model, verbose=False)

    # Append a zero-force strut to ensure it's skipped
    forces.append({
        "i": 0,
        "j": 1,
        "kind": "strut",
        "EA": 0.0,
        "L0": 1.0,
        "L": 1.0,
        "force": 0.0,
    })

    data = buckling_safety_for_struts(model, X, forces, EI=1.0, K=1.0)

    # Original model has 3 struts; the zero-force strut should be skipped
    assert len(data) == 3
    for d in data:
        assert d["safety"] > 0
        assert np.isfinite(d["safety"])
