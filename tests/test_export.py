import numpy as np

from tensegritylab.dr import (
    build_snelson_prism,
    dynamic_relaxation,
    to_member_dataframe,
    to_structure_json,
)


def test_export_dataframe_and_json_consistency():
    model = build_snelson_prism()
    X, forces, _ = dynamic_relaxation(model, tol=1e-6, max_steps=20000, verbose=False)

    df = to_member_dataframe(model, X, forces)
    assert list(df.columns) == ["kind", "i", "j", "L", "force"]
    assert len(df) == len(model.members)
    lengths = [np.linalg.norm(X[m["j"]] - X[m["i"]]) for m in model.members]
    assert np.allclose(df["L"], lengths)

    js = to_structure_json(model, X)
    assert np.allclose(js["nodes"], X)
    assert js["fixed"] == model.fixed.tolist()
    assert df["i"].tolist() == [m["i"] for m in js["members"]]
    assert df["j"].tolist() == [m["j"] for m in js["members"]]
    assert df["kind"].tolist() == [m["kind"] for m in js["members"]]
