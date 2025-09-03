import numpy as np

from tensegritylab.presets import build_snelson_prism, build_quadruple_prism
from tensegritylab.dr import TensegrityModel, dynamic_relaxation, to_structure_json


def test_json_round_trip():
    model = build_snelson_prism()
    X, forces, _ = dynamic_relaxation(model, tol=1e-6, max_steps=20000, verbose=False)
    js = to_structure_json(model, X)
    model2 = TensegrityModel(js["nodes"], js["members"], js["fixed"])
    js2 = to_structure_json(model2, np.array(js["nodes"]))
    assert js == js2


def test_quadruple_prism_basic():
    model = build_quadruple_prism()
    assert model.X.shape == (8, 3)
    struts = [m for m in model.members if m["kind"] == "strut"]
    assert len(struts) == 4
    assert model.fixed.sum() == 4
