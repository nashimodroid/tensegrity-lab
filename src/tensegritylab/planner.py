from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Set

import numpy as np

from .dr import dynamic_relaxation


@dataclass
class Node:
    id: int
    xyz: np.ndarray


@dataclass
class Member:
    i: int
    j: int
    kind: str
    EA: float
    L0: float


@dataclass
class Model:
    nodes: List[Node]
    members: List[Member]
    fixed: Set[int]


def _unique_nodes(struts: Sequence[Sequence[Sequence[float]]]) -> tuple[list[Node], list[tuple[int, int]]]:
    mapping: dict[tuple[float, float, float], int] = {}
    nodes: list[Node] = []
    strut_pairs: list[tuple[int, int]] = []
    for seg in struts:
        pair = []
        for xyz in seg:
            key = tuple(map(float, xyz))
            if key not in mapping:
                nid = len(nodes)
                mapping[key] = nid
                nodes.append(Node(nid, np.array(key, dtype=float)))
            pair.append(mapping[key])
        strut_pairs.append(tuple(pair))
    return nodes, strut_pairs


def _layer_groups(nodes: Sequence[Node]) -> dict[int, list[int]]:
    z = np.array([n.xyz[2] for n in nodes])
    if np.ptp(z) < 1e-9:
        return {0: [n.id for n in nodes]}
    threshold = float(np.median(z))
    low = [n.id for n in nodes if n.xyz[2] <= threshold]
    high = [n.id for n in nodes if n.xyz[2] > threshold]
    return {0: low, 1: high}


def _ring_cables(nodes: Sequence[Node], layer: Sequence[int], edges_set: set[tuple[int, int]], EA: float, L0_scale: float) -> list[Member]:
    pts = np.array([nodes[i].xyz for i in layer])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = [math.atan2(nodes[i].xyz[1]-cy, nodes[i].xyz[0]-cx) for i in layer]
    order = [i for _, i in sorted(zip(angles, layer))]
    members: list[Member] = []
    for a, b in zip(order, order[1:] + order[:1]):
        key = tuple(sorted((a, b)))
        if key in edges_set:
            continue
        L = np.linalg.norm(nodes[a].xyz - nodes[b].xyz)
        members.append(Member(a, b, "cable", EA, L0_scale * L))
        edges_set.add(key)
    return members


def _strut_info(nodes: Sequence[Node], strut_pairs: Sequence[tuple[int, int]]):
    info = []
    pts = np.array([nodes[i].xyz for i in range(len(nodes))])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    for i, j in strut_pairs:
        zb, zt = nodes[i].xyz[2], nodes[j].xyz[2]
        if zb < zt:
            bottom, top = i, j
        else:
            bottom, top = j, i
        xb, yb = nodes[bottom].xyz[:2]
        ang = math.atan2(yb - cy, xb - cx)
        info.append({"bottom": bottom, "top": top, "angle": ang})
    info.sort(key=lambda d: d["angle"])
    return info


def plan_cables_from_struts(
    struts: Sequence[Sequence[Sequence[float]]],
    *,
    fix: str = "auto",
    layer_method: str = "z-kmeans",
    ring: str = "convex_hull",
    cross: str = "nearest_cw",
    side: str = "vertical",
    EA_cable: float = 1.0,
    EA_strut: float = 10.0,
    L0_scale_cable: float = 0.95,
    L0_scale_strut: float = 1.08,
    n_fix: int = 3,
) -> Model:
    nodes, strut_pairs = _unique_nodes(struts)

    members: list[Member] = []
    strut_edges = set()
    for i, j in strut_pairs:
        L = np.linalg.norm(nodes[i].xyz - nodes[j].xyz)
        members.append(Member(i, j, "strut", EA_strut, L0_scale_strut * L))
        strut_edges.add(tuple(sorted((i, j))))

    fixed: Set[int] = set()
    if fix == "auto":
        low_nodes = sorted(nodes, key=lambda n: n.xyz[2])[: n_fix]
        fixed = {n.id for n in low_nodes}

    layers = _layer_groups(nodes)
    edges_set = set(strut_edges)
    for layer in layers.values():
        members.extend(_ring_cables(nodes, layer, edges_set, EA_cable, L0_scale_cable))

    info = _strut_info(nodes, strut_pairs)
    m = len(info)
    for k in range(m):
        s = info[k]
        sn = info[(k + 1) % m]
        # cross cables
        a, b = s["top"], sn["bottom"]
        key = tuple(sorted((a, b)))
        if key not in edges_set and key not in strut_edges:
            L = np.linalg.norm(nodes[a].xyz - nodes[b].xyz)
            members.append(Member(a, b, "cable", EA_cable, L0_scale_cable * L))
            edges_set.add(key)
        # side cables
        a, b = s["bottom"], s["top"]
        key = tuple(sorted((a, b)))
        if key not in edges_set and key not in strut_edges:
            L = np.linalg.norm(nodes[a].xyz - nodes[b].xyz)
            members.append(Member(a, b, "cable", EA_cable, L0_scale_cable * L))
            edges_set.add(key)
        a, b = s["bottom"], info[(k - 1) % m]["top"]
        key = tuple(sorted((a, b)))
        if key not in edges_set and key not in strut_edges:
            L = np.linalg.norm(nodes[a].xyz - nodes[b].xyz)
            members.append(Member(a, b, "cable", EA_cable, L0_scale_cable * L))
            edges_set.add(key)

    return Model(nodes, members, fixed)


def tune_prestress(model: Model, targets: dict) -> None:
    """Adjust member rest lengths to meet force targets."""

    from scipy.optimize import minimize

    base = [m.L0 for m in model.members]
    t_min, t_max = targets.get("cable", (0.0, np.inf))

    def objective(x):
        sc, ss = x
        for m, L0 in zip(model.members, base):
            if m.kind == "cable":
                m.L0 = L0 * sc
            else:
                m.L0 = L0 * ss
        _, forces, _ = dynamic_relaxation(model, verbose=False, tol=1e-4, max_steps=5000)
        pen = 0.0
        for f, m in zip(forces, model.members):
            force = f["force"]
            if m.kind == "cable":
                if force < t_min:
                    pen += (t_min - force) ** 2
                if force > t_max:
                    pen += (force - t_max) ** 2
            else:
                if force > 0:
                    pen += force**2
        return pen

    res = minimize(objective, x0=[1.0, 1.0], bounds=[(0.5, 1.5), (0.5, 1.5)])
    objective(res.x)
    return res


__all__ = [
    "Node",
    "Member",
    "Model",
    "plan_cables_from_struts",
    "tune_prestress",
]
