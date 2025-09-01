from tensegritylab.dr import build_snelson_prism, dynamic_relaxation


def test_fdm_initialization_reduces_steps():
    model = build_snelson_prism()
    _, _, info_plain = dynamic_relaxation(
        model, tol=1e-6, max_steps=20000, verbose=False, use_fdm=False
    )
    _, _, info_fdm = dynamic_relaxation(
        model, tol=1e-6, max_steps=20000, verbose=False, use_fdm=True
    )
    assert info_fdm["steps"] < info_plain["steps"]
