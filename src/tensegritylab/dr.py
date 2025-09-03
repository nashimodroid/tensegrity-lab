import numpy as np

from .presets import build_snelson_prism


class TensegrityModel:
    """Simple container for tensegrity structures."""

    def __init__(self, X, members, fixed=None):
        self.X = np.asarray(X, dtype=float).copy()
        self.members = list(members)
        self.N = self.X.shape[0]
        self.fixed = (
            np.zeros(self.N, dtype=bool)
            if fixed is None
            else np.asarray(fixed, dtype=bool).copy()
        )


def _as_model(model):
    """Coerce various lightweight model containers to ``TensegrityModel``."""

    if isinstance(model, TensegrityModel):
        return model

    if hasattr(model, "nodes") and hasattr(model, "members"):
        X = np.asarray([n.xyz for n in model.nodes], dtype=float)
        fixed = np.zeros(len(model.nodes), dtype=bool)
        for i in getattr(model, "fixed", []):
            fixed[int(i)] = True
        members = [
            {
                "i": int(m.i),
                "j": int(m.j),
                "kind": m.kind,
                "EA": float(m.EA),
                "L0": float(m.L0),
            }
            for m in model.members
        ]
        return TensegrityModel(X, members, fixed)

    raise TypeError("Unsupported model type")


from .presets import build_snelson_prism


def dynamic_relaxation(
    model,
    mass: float = 1.0,
    dt: float = 0.03,
    damping: float = 0.02,
    max_steps: int = 20000,
    tol: float = 1e-6,
    g: float | None = None,
    verbose: bool = True,
    callback=None,
    use_fdm: bool = False,
):
    """Dynamic relaxation solver.

    Parameters
    ----------
    model: TensegrityModel
        Structure definition.
    mass, dt, damping: float
        Physical parameters.
    max_steps: int
        Maximum iterations.
    tol: float
        Convergence tolerance on residual RMS.
    g: float | None
        Gravitational acceleration (positive value acts in -Z).
    verbose: bool
        Print convergence message when True.
    callback: callable(step, rms) | None
        Called every iteration with current step and RMS.
    use_fdm: bool
        When ``True`` the initial coordinates are obtained from the
        Force Density Method instead of using ``model.X``.
    """

    model = _as_model(model)

    if use_fdm:
        from .fdm import fdm_initialize

        q_cable = [m["EA"] / m["L0"] for m in model.members if m["kind"] == "cable"]
        X = fdm_initialize(model, q_cable, fixed=model.fixed)
    else:
        X = model.X.copy()
    V = np.zeros_like(X)
    M = np.full((model.N, 1), mass, dtype=float)
    Pext = np.zeros_like(X)
    if g is not None:
        Pext[:, 2] -= M[:, 0] * g  # gravity along -Z

    fixed = model.fixed
    free = ~fixed

    def assemble_forces(X):
        F = np.zeros_like(X)
        out = []
        for m in model.members:
            i, j = m["i"], m["j"]
            d = X[j] - X[i]
            L = np.linalg.norm(d) + 1e-12
            u = d / L
            f = m["EA"] * (L - m["L0"]) / m["L0"]  # Hooke
            if m["kind"] == "cable":
                f = max(f, 0.0)  # tension only
            else:  # strut
                f = min(f, 0.0)  # compression only
            Fi = f * u
            F[i] += Fi
            F[j] += -Fi
            out.append((L, f, m))
        return F, out

    last_r = 1e9
    step = 0
    for step in range(1, max_steps + 1):
        Fint, _ = assemble_forces(X)
        Fres = Fint + Pext
        Fres[fixed] = 0.0
        rms = np.sqrt((Fres[free] ** 2).sum() / max(1, free.sum()))
        last_r = rms
        if callback is not None:
            callback(step, rms)
        if rms < tol:
            if verbose:
                print(f"[DR] Converged at step {step}, RMS={rms:.3e}")
            break
        A = Fres / M
        V[free] = (1.0 - damping) * V[free] + dt * A[free]
        X[free] = X[free] + dt * V[free]

    _, forces_raw = assemble_forces(X)
    forces = [{**m, "L": L, "force": f} for (L, f, m) in forces_raw]
    return X, forces, {"rms": last_r, "steps": step}


def to_member_dataframe(model, Xf, forces):
    """Export member forces to a :class:`pandas.DataFrame`.

    Parameters
    ----------
    model : TensegrityModel
        The structure definition.
    Xf : array_like
        Final node coordinates after relaxation.
    forces : sequence of dict
        Output from :func:`dynamic_relaxation` describing member forces.
    """

    import pandas as pd

    rows = []
    for m, f in zip(model.members, forces):
        i, j = m["i"], m["j"]
        L = float(np.linalg.norm(Xf[j] - Xf[i]))
        rows.append({"kind": m["kind"], "i": i, "j": j, "L": L, "force": f["force"]})
    return pd.DataFrame(rows, columns=["kind", "i", "j", "L", "force"])


def buckling_safety_for_struts(model, Xf, forces, EI, K=1.0):
    """Compute Euler buckling safety factors for struts.

    Parameters
    ----------
    model : TensegrityModel
        Structure definition (unused but kept for API symmetry).
    Xf : array_like
        Final node coordinates.
    forces : sequence of dict
        Member forces as returned by :func:`dynamic_relaxation`.
    EI : float
        Flexural rigidity of the struts.
    K : float, optional
        Effective length factor.

    Returns
    -------
    list of dict
        Each dict contains ``i``, ``j``, ``L``, ``force``, ``Pcr``, and ``safety``.
        Members with zero or tensile force are skipped.
    """

    out = []
    for m in forces:
        if m["kind"] != "strut":
            continue
        force = m["force"]
        if force >= 0.0 or abs(force) < 1e-12:
            continue
        i, j = m["i"], m["j"]
        L = float(np.linalg.norm(Xf[j] - Xf[i]))
        Pcr = (np.pi**2 * EI) / ((K * L) ** 2)
        safety = Pcr / abs(force)
        out.append({"i": i, "j": j, "L": L, "force": force, "Pcr": Pcr, "safety": safety})
    return out


def to_structure_json(model, Xf):
    """Serialize a model and coordinates to a plain Python ``dict``."""

    return {
        "nodes": Xf.tolist(),
        "members": [
            {"i": m["i"], "j": m["j"], "kind": m["kind"], "EA": m["EA"], "L0": m["L0"]}
            for m in model.members
        ],
        "fixed": model.fixed.tolist(),
    }


def degree_check(model):
    """Check node connectivity degrees.

    Returns a tuple ``(ok, warnings)`` where ``ok`` is ``True`` when every
    node has degree >=1 and a warning list is otherwise provided.
    """

    m = _as_model(model)
    deg = np.zeros(m.N, dtype=int)
    for mem in m.members:
        deg[mem["i"]] += 1
        deg[mem["j"]] += 1
    warnings = []
    if np.any(deg < 1):
        warnings.append("isolated nodes detected")
    if np.any(deg < 3):
        warnings.append("nodes with degree < 3 present")
    return len(warnings) == 0, warnings
__all__ = [
    "build_snelson_prism",
    "dynamic_relaxation",
    "buckling_safety_for_struts",
    "to_member_dataframe",
    "to_structure_json",
    "degree_check",
]
