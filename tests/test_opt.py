import pandas as pd

from tensegritylab.presets import build_snelson_prism
from tensegritylab.opt import sweep_prestress


def test_sweep_prestress_history_and_best():
    model = build_snelson_prism()
    cable_scales = [0.95, 1.0]
    strut_scales = [1.0, 1.05]
    best, history = sweep_prestress(model, cable_scales, strut_scales)
    assert isinstance(best, dict)
    assert isinstance(history, pd.DataFrame)
    assert len(history) == len(cable_scales) * len(strut_scales)
    assert {"cable_L0_scale", "strut_L0_scale", "score"}.issubset(best.keys())
