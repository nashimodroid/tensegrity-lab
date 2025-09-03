import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tensegritylab.presets import build_snelson_prism, build_quadruple_prism
from tensegritylab.dr import (
    dynamic_relaxation,
    buckling_safety_for_struts,
    to_member_dataframe,
    to_structure_json,
    TensegrityModel,
)
from tensegritylab.opt import sweep_prestress
from tensegritylab.planner import plan_cables_from_struts
from tensegritylab.editor_state import add_strut, edit_strut, delete_strut

st.title("Tensegrity Dynamic Relaxation")

mode = st.radio("Mode", ["Preset", "Custom editor"], horizontal=True)

if mode == "Custom editor":
    st.header("Strut Editor")
    struts = st.session_state.setdefault("struts", [])
    selected = st.session_state.setdefault("selected_strut", 0)

    col1, _, col3 = st.columns(3)
    if col1.button("Add"):
        struts = add_strut(struts, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        st.session_state["struts"] = struts
        st.session_state["selected_strut"] = len(struts) - 1

    if struts:
        idx = st.selectbox(
            "Select strut", list(range(len(struts))), index=min(selected, len(struts) - 1)
        )
        st.session_state["selected_strut"] = idx
        a, b = struts[idx]
        ax = st.slider("Ax", -2.0, 2.0, float(a[0]), step=0.05)
        ay = st.slider("Ay", -2.0, 2.0, float(a[1]), step=0.05)
        az = st.slider("Az", -2.0, 2.0, float(a[2]), step=0.05)
        bx = st.slider("Bx", -2.0, 2.0, float(b[0]), step=0.05)
        by = st.slider("By", -2.0, 2.0, float(b[1]), step=0.05)
        bz = st.slider("Bz", -2.0, 2.0, float(b[2]), step=0.05)
        struts = edit_strut(struts, idx, (ax, ay, az), (bx, by, bz))
        st.session_state["struts"] = struts

        if col3.button("Delete"):
            struts = delete_strut(struts, idx)
            st.session_state["struts"] = struts
            st.session_state["selected_strut"] = 0

    df = pd.DataFrame(
        [s[0] + s[1] for s in st.session_state["struts"]],
        columns=["x0", "y0", "z0", "x1", "y1", "z1"],
    )
    edited_df = st.data_editor(df, num_rows="dynamic")
    st.session_state["struts"] = [
        ((row.x0, row.y0, row.z0), (row.x1, row.y1, row.z1))
        for row in edited_df.itertuples(index=False)
    ]

    fig = go.Figure()
    sx, sy, sz = [], [], []
    nx, ny, nz = [], [], []
    for a, b in st.session_state["struts"]:
        sx.extend([a[0], b[0], None])
        sy.extend([a[1], b[1], None])
        sz.extend([a[2], b[2], None])
        nx.extend([a[0], b[0]])
        ny.extend([a[1], b[1]])
        nz.extend([a[2], b[2]])
    if st.session_state["struts"]:
        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Strut",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=nx,
                y=ny,
                z=nz,
                mode="markers",
                marker=dict(color="black"),
                name="Nodes",
            )
        )
    fig.update_layout(scene=dict(aspectmode="data"))
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Auto-plan Cables"):
        model = plan_cables_from_struts(st.session_state["struts"])
        st.session_state["planned_model"] = model

    if "planned_model" in st.session_state:
        model = st.session_state["planned_model"]
        X = np.array([n.xyz for n in model.nodes])
        cable_x, cable_y, cable_z = [], [], []
        strut_x, strut_y, strut_z = [], [], []
        for m in model.members:
            i, j = m.i, m.j
            xs = [X[i, 0], X[j, 0], None]
            ys = [X[i, 1], X[j, 1], None]
            zs = [X[i, 2], X[j, 2], None]
            if m.kind == "cable":
                cable_x.extend(xs)
                cable_y.extend(ys)
                cable_z.extend(zs)
            else:
                strut_x.extend(xs)
                strut_y.extend(ys)
                strut_z.extend(zs)
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter3d(
                x=cable_x,
                y=cable_y,
                z=cable_z,
                mode="lines",
                line=dict(color="blue"),
                name="Cable",
            )
        )
        fig2.add_trace(
            go.Scatter3d(
                x=strut_x,
                y=strut_y,
                z=strut_z,
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Strut",
            )
        )
        fig2.add_trace(
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode="markers",
                marker=dict(color="lightgray"),
                name="Nodes",
            )
        )
        fig2.update_layout(scene=dict(aspectmode="data"))
        st.plotly_chart(fig2, use_container_width=True)

        use_fdm = st.checkbox("Use FDM initialization", value=False, key="use_fdm_custom")
        if st.button("Solve"):
            progress = st.empty()

            def cb(step, rms):
                progress.text(f"step {step}: RMS={rms:.2e}")

            with st.spinner("Solving..."):
                X, forces, info = dynamic_relaxation(
                    model, callback=cb, verbose=False, use_fdm=use_fdm
                )

            progress.text(
                f"Converged in {info['steps']} steps, RMS={info['rms']:.2e}"
            )
            fig3 = go.Figure()
            cable_x, cable_y, cable_z = [], [], []
            strut_x, strut_y, strut_z = [], [], []
            for m in model.members:
                i, j = m.i, m.j
                xs = [X[i, 0], X[j, 0], None]
                ys = [X[i, 1], X[j, 1], None]
                zs = [X[i, 2], X[j, 2], None]
                if m.kind == "cable":
                    cable_x.extend(xs)
                    cable_y.extend(ys)
                    cable_z.extend(zs)
                else:
                    strut_x.extend(xs)
                    strut_y.extend(ys)
                    strut_z.extend(zs)
            fig3.add_trace(
                go.Scatter3d(
                    x=cable_x,
                    y=cable_y,
                    z=cable_z,
                    mode="lines",
                    line=dict(color="blue"),
                    name="Cable",
                )
            )
            fig3.add_trace(
                go.Scatter3d(
                    x=strut_x,
                    y=strut_y,
                    z=strut_z,
                    mode="lines",
                    line=dict(color="red"),
                    name="Strut",
                )
            )
            fig3.add_trace(
                go.Scatter3d(
                    x=X[:, 0],
                    y=X[:, 1],
                    z=X[:, 2],
                    mode="markers",
                    marker=dict(color="lightgray"),
                    name="Nodes",
                )
            )
            fig3.update_layout(scene=dict(aspectmode="data"), showlegend=True)
            st.plotly_chart(fig3, use_container_width=True)

            df = to_member_dataframe(model, X, forces)
            st.dataframe(df)

    st.stop()

uploaded = st.file_uploader("Load JSON", type="json")
default_idx = 2 if uploaded else 0
preset = st.selectbox("Preset", ["Prism3", "Prism4", "Custom"], index=default_idx)

radius = st.slider("radius", 0.5, 2.0, 1.0, step=0.1)
height = st.slider("height", 0.5, 2.0, 1.2, step=0.1)
twist_deg = st.slider("twist_deg", 0.0, 180.0, 150.0, step=1.0)
EA_cable = st.slider("EA_cable", 0.1, 10.0, 1.0, step=0.1)
EA_strut = st.slider("EA_strut", 1.0, 50.0, 10.0, step=1.0)
cable_L0_scale = st.slider("cable_L0_scale", 0.5, 1.0, 0.95, step=0.01)
strut_L0_scale = st.slider("strut_L0_scale", 1.0, 1.5, 1.2, step=0.01)
EI = st.number_input("EI", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
K = st.number_input("K", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
use_fdm = st.checkbox("Use FDM initialization", value=False)

loaded_model = None
if uploaded is not None:
    data = json.load(uploaded)
    loaded_model = TensegrityModel(data["nodes"], data["members"], data.get("fixed"))

if st.button("Solve"):
    if preset == "Prism3":
        model = build_snelson_prism(
            r=radius,
            h=height,
            theta=np.deg2rad(twist_deg),
            EA_cable=EA_cable,
            EA_strut=EA_strut,
            cable_L0_scale=cable_L0_scale,
            strut_L0_scale=strut_L0_scale,
        )
    elif preset == "Prism4":
        model = build_quadruple_prism(
            r=radius,
            h=height,
            theta=np.deg2rad(twist_deg),
            EA_cable=EA_cable,
            EA_strut=EA_strut,
            cable_L0_scale=cable_L0_scale,
            strut_L0_scale=strut_L0_scale,
        )
    else:
        if loaded_model is None:
            st.error("Upload a JSON for Custom preset")
            st.stop()
        model = loaded_model
    progress = st.empty()

    def cb(step, rms):
        progress.text(f"step {step}: RMS={rms:.2e}")

    with st.spinner("Solving..."):
        X, forces, info = dynamic_relaxation(
            model, callback=cb, verbose=False, use_fdm=use_fdm
        )

    progress.text(
        f"Converged in {info['steps']} steps, RMS={info['rms']:.2e}"
    )
    cols = st.columns(2)
    cols[0].metric("Steps", info["steps"])
    cols[1].metric("RMS", f"{info['rms']:.2e}")
    if use_fdm:
        st.caption("Initialized by FDM")

    fig = go.Figure()
    cable_x, cable_y, cable_z = [], [], []
    strut_x, strut_y, strut_z = [], [], []
    for m in model.members:
        i, j = m["i"], m["j"]
        xs = [X[i, 0], X[j, 0], None]
        ys = [X[i, 1], X[j, 1], None]
        zs = [X[i, 2], X[j, 2], None]
        if m["kind"] == "cable":
            cable_x.extend(xs)
            cable_y.extend(ys)
            cable_z.extend(zs)
        else:
            strut_x.extend(xs)
            strut_y.extend(ys)
            strut_z.extend(zs)

    fig.add_trace(
        go.Scatter3d(
            x=cable_x,
            y=cable_y,
            z=cable_z,
            mode="lines",
            line=dict(color="blue"),
            name="Cable",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=strut_x,
            y=strut_y,
            z=strut_z,
            mode="lines",
            line=dict(color="red"),
            name="Strut",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(color="lightgray"),
            name="Nodes",
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    df = to_member_dataframe(model, X, forces)
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "forces.csv", "text/csv")

    buckling = buckling_safety_for_struts(model, X, forces, EI, K)
    buckling_df = pd.DataFrame(buckling)
    if not buckling_df.empty:
        styler = buckling_df.style.applymap(
            lambda v: "background-color: #faa" if v < 1.5 else "",
            subset=["safety"],
        )
        st.dataframe(styler)
        buckling_csv = buckling_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Buckling CSV", buckling_csv, "buckling.csv", "text/csv"
        )

    js = to_structure_json(model, X)
    js_str = json.dumps(js)
    st.download_button(
        "Save JSON", js_str, "structure.json", "application/json"
    )


st.header("Optimization")

col1, col2, col3 = st.columns(3)
with col1:
    c_min = st.number_input("cable_L0_scale min", 0.5, 1.0, 0.9, step=0.01)
with col2:
    c_max = st.number_input("cable_L0_scale max", 0.5, 1.0, 1.0, step=0.01)
with col3:
    c_step = st.number_input("cable_L0_scale step", 0.01, 0.5, 0.02, step=0.01)

col4, col5, col6 = st.columns(3)
with col4:
    s_min = st.number_input("strut_L0_scale min", 1.0, 2.0, 1.0, step=0.01)
with col5:
    s_max = st.number_input("strut_L0_scale max", 1.0, 2.0, 1.2, step=0.01)
with col6:
    s_step = st.number_input("strut_L0_scale step", 0.01, 1.0, 0.05, step=0.01)

metric = st.selectbox("metric", ["min_tension_spread", "max_buckling_safety"])

if st.button("Run sweep"):
    if preset == "Prism3":
        base = build_snelson_prism(
            r=radius,
            h=height,
            theta=np.deg2rad(twist_deg),
            EA_cable=EA_cable,
            EA_strut=EA_strut,
            cable_L0_scale=1.0,
            strut_L0_scale=1.0,
        )
    elif preset == "Prism4":
        base = build_quadruple_prism(
            r=radius,
            h=height,
            theta=np.deg2rad(twist_deg),
            EA_cable=EA_cable,
            EA_strut=EA_strut,
            cable_L0_scale=1.0,
            strut_L0_scale=1.0,
        )
    else:
        if loaded_model is None:
            st.error("Upload a JSON for Custom preset")
            st.stop()
        base = loaded_model

    cable_scales = np.arange(c_min, c_max + 1e-12, c_step)
    strut_scales = np.arange(s_min, s_max + 1e-12, s_step)
    best, history = sweep_prestress(base, cable_scales, strut_scales, metric=metric)
    st.write("Best parameters", best)
    st.dataframe(history)
    hist_csv = history.to_csv(index=False).encode("utf-8")
    st.download_button("Download sweep CSV", hist_csv, "sweep.csv", "text/csv")
