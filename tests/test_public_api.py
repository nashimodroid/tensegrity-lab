from tensegritylab.dr import build_snelson_prism, dynamic_relaxation


def test_snelson_prism_converges():
    model = build_snelson_prism()
    _, _, info = dynamic_relaxation(model, tol=1e-6, max_steps=20000, verbose=False)
    assert info["rms"] < 1e-6
