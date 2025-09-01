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
