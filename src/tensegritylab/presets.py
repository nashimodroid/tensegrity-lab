import numpy as np


def _build_prism(
    n: int,
    r: float = 1.0,
    h: float = 1.2,
    theta: float = np.deg2rad(150),
    EA_cable: float = 1.0,
    EA_strut: float = 10.0,
    cable_L0_scale: float = 0.95,
    strut_L0_scale: float = 1.20,
) -> "TensegrityModel":
    """Internal helper to create an n-sided prism."""

    from .dr import TensegrityModel

    B = np.array(
        [[r * np.cos(2 * np.pi * k / n), r * np.sin(2 * np.pi * k / n), 0.0] for k in range(n)]
    )
    T = np.array(
        [
            [
                r * np.cos(2 * np.pi * k / n + theta),
                r * np.sin(2 * np.pi * k / n + theta),
                h,
            ]
            for k in range(n)
        ]
    )
    X0 = np.vstack([B, T])

    members: list[dict] = []

    def add(i: int, j: int, kind: str, EA: float, L0_scale: float) -> None:
        L = np.linalg.norm(X0[j] - X0[i])
        members.append({"i": i, "j": j, "kind": kind, "EA": EA, "L0": L0_scale * L})

    for k in range(n):
        add(k, n + ((k + 1) % n), "strut", EA_strut, strut_L0_scale)

    for k in range(n):
        add(k, (k + 1) % n, "cable", EA_cable, cable_L0_scale)
        add(n + k, n + ((k + 1) % n), "cable", EA_cable, cable_L0_scale)
        add(k, n + k, "cable", EA_cable, cable_L0_scale)
        add(k, n + ((k - 1) % n), "cable", EA_cable, cable_L0_scale)

    fixed = np.zeros(2 * n, dtype=bool)
    fixed[:n] = True
    return TensegrityModel(X0, members, fixed)


def build_snelson_prism(
    r: float = 1.0,
    h: float = 1.2,
    theta: float = np.deg2rad(150),
    EA_cable: float = 1.0,
    EA_strut: float = 10.0,
    cable_L0_scale: float = 0.95,
    strut_L0_scale: float = 1.20,
) -> "TensegrityModel":
    """Create a 3-strut Snelson prism model."""

    return _build_prism(
        3,
        r=r,
        h=h,
        theta=theta,
        EA_cable=EA_cable,
        EA_strut=EA_strut,
        cable_L0_scale=cable_L0_scale,
        strut_L0_scale=strut_L0_scale,
    )


def build_quadruple_prism(
    r: float = 1.0,
    h: float = 1.2,
    theta: float = np.deg2rad(135),
    EA_cable: float = 1.0,
    EA_strut: float = 10.0,
    cable_L0_scale: float = 0.95,
    strut_L0_scale: float = 1.20,
) -> "TensegrityModel":
    """Create a 4-strut prism model."""

    return _build_prism(
        4,
        r=r,
        h=h,
        theta=theta,
        EA_cable=EA_cable,
        EA_strut=EA_strut,
        cable_L0_scale=cable_L0_scale,
        strut_L0_scale=strut_L0_scale,
    )


def build_quintuple_prism(
    r: float = 1.0,
    h: float = 1.2,
    theta: float = np.deg2rad(140),
    EA_cable: float = 1.0,
    EA_strut: float = 10.0,
    cable_L0_scale: float = 0.95,
    strut_L0_scale: float = 1.20,
) -> "TensegrityModel":
    """Create a 5-strut prism model."""

    return _build_prism(
        5,
        r=r,
        h=h,
        theta=theta,
        EA_cable=EA_cable,
        EA_strut=EA_strut,
        cable_L0_scale=cable_L0_scale,
        strut_L0_scale=strut_L0_scale,
    )


__all__ = [
    "build_snelson_prism",
    "build_quadruple_prism",
    "build_quintuple_prism",
]
