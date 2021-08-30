import streamlit as st
import networkx as nx
import minitorch
from minitorch import MathTest, MathTestVariable
import plotly.graph_objects as go
import inspect
import graph_builder

MyModule = None
minitorch


def render_function(fn):
    st.markdown(
        """
```python
%s

```"""
        % inspect.getsource(fn)
    )


def render_math_sandbox(use_scalar=False):
    st.write("# Sandbox for the Math Functions")

    if use_scalar:
        one, two, red = MathTestVariable._tests()
    else:
        one, two, red = MathTest._tests()
    f_type = st.selectbox("Function Type", ["One Arg", "Two Arg", "Reduce"])
    select = {"One Arg": one, "Two Arg": two, "Reduce": red}

    fn = st.selectbox("Function", select[f_type], format_func=lambda a: a[0])
    name, _, scalar = fn
    if f_type == "One Arg":
        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x)")
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        if use_scalar:
            ys = [scalar(minitorch.Scalar(p)).data for p in xs]
        else:
            ys = [scalar(p) for p in xs]
        scatter = go.Scatter(mode="lines", x=xs, y=ys)
        fig = go.Figure(scatter)
        st.write(fig)

        if use_scalar:
            st.write("Derivative f'(x)")
            x_var = [minitorch.Scalar(x) for x in xs]
            for x in x_var:
                out = scalar(x)
                out.backward()
            scatter = go.Scatter(mode="lines", x=xs, y=[x.derivative for x in x_var])
            fig = go.Figure(scatter)
            st.write(fig)
            G = graph_builder.GraphBuilder().run(out)
            G.graph["graph"] = {"rankdir": "LR"}
            st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

    if f_type == "Two Arg":

        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x, y)")
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        if use_scalar:
            zs = [
                [scalar(minitorch.Scalar(x), minitorch.Scalar(y)).data for x in xs]
                for y in ys
            ]
        else:
            zs = [[scalar(x, y) for x in xs] for y in ys]
        scatter = go.Surface(x=xs, y=ys, z=[zs for y in ys])

        fig = go.Figure(scatter)
        st.write(fig)

        if use_scalar:
            a, b = [], []
            for x in xs:
                oa, ob = [], []
                for y in ys:
                    x1 = minitorch.Scalar(x)
                    y1 = minitorch.Scalar(y)
                    out = scalar(x1, y1)
                    out.backward()
                    oa.append((x, y, x1.derivative))
                    ob.append((x, y, y1.derivative))
                a.append(oa)
                b.append(ob)
            st.write("Derivative f'_x(x, y)")

            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in a],
                y=[[c[1] for c in a2] for a2 in a],
                z=[[c[2] for c in a2] for a2 in a],
            )
            fig = go.Figure(scatter)
            st.write(fig)
            st.write("Derivative f'_y(x, y)")
            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in b],
                y=[[c[1] for c in a2] for a2 in b],
                z=[[c[2] for c in a2] for a2 in b],
            )
            fig = go.Figure(scatter)
            st.write(fig)
    if f_type == "Reduce":
        st.write("### " + name)
        render_function(scalar)
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]

        scatter = go.Surface(x=xs, y=ys, z=[[scalar([x, y]) for x in xs] for y in ys])
        fig = go.Figure(scatter)
        st.write(fig)
