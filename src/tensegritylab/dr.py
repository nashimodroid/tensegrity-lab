import numpy as np


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


def dynamic_relaxation(
    model: TensegrityModel,
    mass: float = 1.0,
    dt: float = 0.02,
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
    """

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
        if callback is not None:
            callback(step, rms)
        if rms < tol:
            if verbose:
                print(f"[DR] Converged at step {step}, RMS={rms:.3e}")
            break
        A = Fres / M
        V[free] = (1.0 - damping) * V[free] + dt * A[free]
        X[free] = X[free] + dt * V[free]
        last_r = rms

    _, forces_raw = assemble_forces(X)
    forces = [{**m, "L": L, "force": f} for (L, f, m) in forces_raw]
    return X, forces, {"rms": last_r, "steps": step}


def build_snelson_prism(
    r: float = 1.0,
    h: float = 1.2,
    theta: float = np.deg2rad(150),
    EA_cable: float = 1.0,
    EA_strut: float = 10.0,
    cable_L0_scale: float = 0.95,
    strut_L0_scale: float = 1.20,
):
    """Create a 3-strut Snelson prism model.

    Parameters are mostly geometric and material properties.
    """

    B = np.array(
        [[r * np.cos(2 * np.pi * k / 3), r * np.sin(2 * np.pi * k / 3), 0.0] for k in range(3)]
    )
    T = np.array(
        [
            [
                r * np.cos(2 * np.pi * k / 3 + theta),
                r * np.sin(2 * np.pi * k / 3 + theta),
                h,
            ]
            for k in range(3)
        ]
    )
    X0 = np.vstack([B, T])

    members = []

    def add(i, j, kind, EA, L0_scale):
        L = np.linalg.norm(X0[j] - X0[i])
        members.append({"i": i, "j": j, "kind": kind, "EA": EA, "L0": L0_scale * L})

    for k in range(3):
        add(k, 3 + ((k + 1) % 3), "strut", EA_strut, strut_L0_scale)

    for (i, j) in [(0, 1), (1, 2), (2, 0)]:
        add(i, j, "cable", EA_cable, cable_L0_scale)
        add(3 + i, 3 + j, "cable", EA_cable, cable_L0_scale)

    for k in range(3):
        add(k, 3 + k, "cable", EA_cable, cable_L0_scale)
        add(k, 3 + ((k - 1) % 3), "cable", EA_cable, cable_L0_scale)

    fixed = np.zeros(6, dtype=bool)
    fixed[:3] = True
    return TensegrityModel(X0, members, fixed)


__all__ = ["build_snelson_prism", "dynamic_relaxation"]
