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

st.title("Tensegrity Dynamic Relaxation")

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
