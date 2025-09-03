import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tensegritylab.dr import (
    build_snelson_prism,
    dynamic_relaxation,
    buckling_safety_for_struts,
    to_member_dataframe,
    to_structure_json,
)

st.title("Tensegrity Dynamic Relaxation")

radius = st.slider("radius", 0.5, 2.0, 1.0)
height = st.slider("height", 0.5, 2.0, 1.2)
twist_deg = st.slider("twist_deg", 0.0, 180.0, 150.0)
EA_cable = st.slider("EA_cable", 0.1, 10.0, 1.0)
EA_strut = st.slider("EA_strut", 1.0, 50.0, 10.0)
cable_L0_scale = st.slider("cable_L0_scale", 0.5, 1.0, 0.95)
strut_L0_scale = st.slider("strut_L0_scale", 1.0, 1.5, 1.2)
EI = st.number_input("EI", value=1.0)
K = st.number_input("K", value=1.0)
use_fdm = st.checkbox("Use FDM initialization", value=False)

if st.button("Solve"):
    model = build_snelson_prism(
        r=radius,
        h=height,
        theta=np.deg2rad(twist_deg),
        EA_cable=EA_cable,
        EA_strut=EA_strut,
        cable_L0_scale=cable_L0_scale,
        strut_L0_scale=strut_L0_scale,
    )
    placeholder = st.empty()

    def cb(step, rms):
        placeholder.write(f"step {step}: RMS={rms:.2e}")

    X, forces, info = dynamic_relaxation(
        model, callback=cb, verbose=False, use_fdm=use_fdm
    )
    placeholder.write(
        f"Converged in {info['steps']} steps, RMS={info['rms']:.2e}"
    )

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
        "Download JSON", js_str, "structure.json", "application/json"
    )
