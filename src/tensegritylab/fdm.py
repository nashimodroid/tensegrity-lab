import numpy as np


def fdm_initialize(model, q_cable, fixed=None):
    """Initialize node positions using the Force Density Method.

    Parameters
    ----------
    model : TensegrityModel
        Structure definition. Only cable members are considered.
    q_cable : float or array-like
        Force densities for cable members. If scalar, the same density is
        applied to all cables. If array-like, it must match the number of
        cable members.
    fixed : array-like of bool, optional
        Boolean mask of fixed nodes. If ``None`` uses ``model.fixed``.

    Returns
    -------
    ndarray
        Node coordinates computed by the FDM. Nodes not connected to any
        fixed node are left unchanged.
    """

    X = np.asarray(model.X, dtype=float)
    N = X.shape[0]

    if fixed is None:
        fixed = model.fixed
    fixed = np.asarray(fixed, dtype=bool)

    # Collect cable members and adjacency for graph traversal
    cables = []
    adj = {i: set() for i in range(N)}
    for m in model.members:
        if m["kind"] != "cable":
            continue
        i, j = m["i"], m["j"]
        cables.append((i, j))
        adj[i].add(j)
        adj[j].add(i)

    # Determine subnet connected to fixed nodes to avoid degeneracy
    if fixed.any():
        visited = set(np.where(fixed)[0])
        queue = list(visited)
        while queue:
            i = queue.pop(0)
            for j in adj[i]:
                if j not in visited:
                    visited.add(j)
                    queue.append(j)
        mask = np.zeros(N, dtype=bool)
        mask[list(visited)] = True
    else:
        mask = np.ones(N, dtype=bool)

    free = ~fixed & mask
    if not np.any(free):
        return X.copy()

    # Force density matrix (Laplacian-like)
    L = np.zeros((N, N))
    if np.isscalar(q_cable):
        q_vals = [float(q_cable)] * len(cables)
    else:
        q_vals = np.asarray(q_cable, dtype=float)
        if q_vals.size != len(cables):
            raise ValueError("q_cable must match number of cable members")
    for (i, j), q in zip(cables, q_vals):
        L[i, i] += q
        L[j, j] += q
        L[i, j] -= q
        L[j, i] -= q

    idx_free = np.where(free)[0]
    idx_fixed = np.where(fixed & mask)[0]
    L_ff = L[np.ix_(idx_free, idx_free)]
    L_fb = L[np.ix_(idx_free, idx_fixed)]
    B = -L_fb @ X[idx_fixed]
    X_sol = np.linalg.solve(L_ff, B)

    X0 = X.copy()
    X0[idx_free] = X_sol
    return X0


__all__ = ["fdm_initialize"]
