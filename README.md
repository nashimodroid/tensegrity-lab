# Tensegrity Lab

A minimal dynamic relaxation solver for tensegrity structures.

## Local run

After installing the package, launch the Streamlit application with:

```bash
streamlit run examples/app.py
```

## FDM 初期化

The Force Density Method (FDM) provides an analytical starting shape by
solving a linear system for the free node coordinates.
For a cable network with force densities $q$, the equilibrium is

\begin{equation}
L_{ff} \mathbf{X}_f = -L_{fb} \mathbf{X}_b,
\end{equation}

where $L$ is assembled from the cable force densities and $\mathbf{X}_b$
are the coordinates of fixed nodes. The solution $\mathbf{X}_f$ is used as
the initial configuration for dynamic relaxation.

Enable this initialization with:

```python
dynamic_relaxation(model, use_fdm=True)
```

This typically reduces the number of iterations required for convergence.
The Streamlit demo provides a *Use FDM initialization* checkbox to toggle
this behaviour interactively.

## 出力

Member forces can be exported to CSV via `to_member_dataframe`, while
`to_structure_json` serializes the geometry and member data to JSON.
The Streamlit demo offers download buttons for both formats.

## 座屈チェック

`buckling_safety_for_struts` evaluates Euler buckling safety factors for
compressive members. Values below a chosen threshold (e.g. 1.5) highlight
struts that may require redesign.
