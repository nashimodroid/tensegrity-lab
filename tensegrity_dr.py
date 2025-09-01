# tensegrity_dr.py
# --- 最小のテンセグリティ DR ソルバ & 例（3本ストラット・プリズム） ---
import numpy as np

class TensegrityModel:
    """
    X: (N,3) ノード座標, members: 部材辞書のリスト,
    fixed: True のノードは固定（変位0）
    """
    def __init__(self, X, members, fixed=None):
        self.X = np.asarray(X, dtype=float).copy()
        self.members = list(members)
        self.N = self.X.shape[0]
        self.fixed = np.zeros(self.N, dtype=bool) if fixed is None else np.asarray(fixed, dtype=bool).copy()

def dynamic_relaxation(model: TensegrityModel,
                       mass=1.0, dt=0.02, damping=0.02,
                       max_steps=20000, tol=1e-6, g=None, verbose=True):
    """
    動的緩和（DR）: ケーブル=張力のみ, ストラット=圧縮のみ
    """
    X = model.X.copy()
    V = np.zeros_like(X)
    M = np.full((model.N, 1), mass, dtype=float)
    Pext = np.zeros_like(X)
    if g is not None:
        Pext[:, 2] -= M[:, 0] * g  # 重力(-Z)

    fixed = model.fixed
    free = ~fixed

    def assemble_forces(X):
        F = np.zeros_like(X)
        out = []
        for m in model.members:
            i, j = m['i'], m['j']
            d = X[j] - X[i]
            L = np.linalg.norm(d) + 1e-12
            u = d / L
            # フック則: 軸力 f = EA * (L - L0) / L0
            f = m['EA'] * (L - m['L0']) / m['L0']
            if m['kind'] == 'cable':
                f = max(f, 0.0)     # 張力のみ許容
            else:                   # 'strut'
                f = min(f, 0.0)     # 圧縮のみ許容
            Fi = f * u
            F[i] +=  Fi
            F[j] += -Fi
            out.append((L, f, m))
        return F, out

    last_r = 1e9
    step = 0
    for step in range(1, max_steps + 1):
        Fint, _ = assemble_forces(X)
        Fres = Fint + Pext
        Fres[fixed] = 0.0  # 拘束反力に吸収
        # 自由節点の残差RMSで収束判定
        rms = np.sqrt((Fres[free]**2).sum() / max(1, free.sum()))
        if rms < tol:
            if verbose:
                print(f"[DR] Converged at step {step}, RMS={rms:.3e}")
            break
        # セミ・インプリシット・オイラーで更新
        A = Fres / M
        V[free] = (1.0 - damping) * V[free] + dt * A[free]
        X[free] = X[free] + dt * V[free]
        last_r = rms

    # 最終状態の部材力を評価
    _, forces_raw = assemble_forces(X)
    forces = [{**m, 'L': L, 'force': f} for (L, f, m) in forces_raw]
    return X, forces, {'rms': last_r, 'steps': step}

def build_snelson_prism(r=1.0, h=1.2, theta=np.deg2rad(150),
                        EA_cable=1.0, EA_strut=10.0):
    """
    3本ストラット・プリズム。下三角を固定。
    - ストラット: b_k -> t_{k+1}（圧縮のみ）
    - リングケーブル: bottom/top 各3本（張力のみ）
    - サイドケーブル: (b_k, t_k), (b_k, t_{k-1})
    L0 は「ケーブル短め」「ストラット長め」に置いてプレストレス状態へ誘導。
    """
    B = np.array([[r*np.cos(2*np.pi*k/3), r*np.sin(2*np.pi*k/3), 0.0] for k in range(3)])
    T = np.array([[r*np.cos(2*np.pi*k/3 + theta), r*np.sin(2*np.pi*k/3 + theta), h] for k in range(3)])
    X0 = np.vstack([B, T])  # 0..2: bottom, 3..5: top

    members = []
    def add(i, j, kind, EA, L0_scale):
        L = np.linalg.norm(X0[j] - X0[i])
        members.append({'i': i, 'j': j, 'kind': kind, 'EA': EA, 'L0': L0_scale * L})

    # ストラット（最終的に圧縮にしたいので L0 は長め = 1.20*初期長）
    for k in range(3):
        add(k, 3 + ((k+1) % 3), 'strut', EA_strut, L0_scale=1.20)

    # リングケーブル（短め）
    for (i, j) in [(0,1),(1,2),(2,0)]:
        add(i, j, 'cable', EA_cable, L0_scale=0.92)        # bottom
        add(3+i, 3+j, 'cable', EA_cable, L0_scale=0.92)    # top

    # サイドケーブル（短め）
    for k in range(3):
        add(k, 3+k, 'cable', EA_cable, L0_scale=0.95)                  # (b_k, t_k)
        add(k, 3+((k-1) % 3), 'cable', EA_cable, L0_scale=0.95)        # (b_k, t_{k-1})

    fixed = np.zeros(6, dtype=bool); fixed[:3] = True
    return TensegrityModel(X0, members, fixed)

# --- おまけ：簡単な3Dプロット（任意、matplotlibが入っていれば表示されます） ---
def plot_model(X, members, title=None):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print("(Viewer skipped: matplotlib not available)")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for m in members:
        i, j = m['i'], m['j']
        xs = [X[i,0], X[j,0]]
        ys = [X[i,1], X[j,1]]
        zs = [X[i,2], X[j,2]]
        ls = '-' if m['kind']=='cable' else '--'
        ax.plot(xs, ys, zs, linestyle=ls, linewidth=2)
    ax.scatter(X[:,0], X[:,1], X[:,2], s=25)
    if title: ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    model = build_snelson_prism()
    Xf, forces, info = dynamic_relaxation(model, dt=0.02, damping=0.02,
                                          tol=1e-7, max_steps=15000, verbose=True)

    print("\n[Result] steps =", info['steps'], "  RMS =", f"{info['rms']:.2e}")
    print("Top nodes (approx):\n", np.round(Xf[3:6], 3))

    cable_forces = [m['force'] for m in forces if m['kind']=='cable']
    strut_forces  = [m['force'] for m in forces if m['kind']=='strut']
    print(f"Cable tension  [min,max] = {min(cable_forces):.3f}, {max(cable_forces):.3f}")
    print(f"Strut compression (negative) [min,max] = {min(strut_forces):.3f}, {max(strut_forces):.3f}")

    # 図示（任意）
    plot_model(Xf, model.members, title="Tensegrity prism (final)")
