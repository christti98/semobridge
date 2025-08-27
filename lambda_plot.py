import plotly.graph_objs as go

# === 1. Plot for lambda_it ===
x_it = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lambda_it_acc = [
    73.91,
    73.89,
    73.94,
    73.90,
    74.01,
    74.07,
    74.07,
    74.08,
    73.96,
    73.73,
]

trace_it = go.Scatter(
    x=x_it,
    y=lambda_it_acc,
    mode="lines+markers",
    name=r"$\lambda_\mathrm{it}$",
    line=dict(color="purple", width=6),
    marker=dict(size=20),
)

layout_single = go.Layout(
    xaxis=dict(
        title=r"$\Huge\lambda_\mathrm{it}$",
        title_standoff=10,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="lightgrey",
        gridwidth=1,
    ),
    yaxis=dict(
        title="Accuracy (%)",
        title_standoff=50,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="lightgrey",
        gridwidth=1,
        range=[72.0, 75],
    ),
    legend=dict(
        x=0.01,
        y=0.99,
        font=dict(size=36, family="Times New Roman"),
    ),
    plot_bgcolor="white",
    font=dict(size=36, family="Times New Roman", color="black"),
    margin=dict(l=5, r=2, t=40, b=120),
    width=1700,
    height=650,
)

fig_it = go.Figure(data=[trace_it], layout=layout_single)
fig_it.write_image("lambda_it_plot.pdf")

# === 2. Combined plot for lambda_c and lambda_b ===
x_cb = [0.05, 0.1, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
lambda_c_acc = [74.01, 73.90, 73.91, 73.92, 73.99, 73.92, 74.02, 74.00, 73.98, 73.96]
lambda_b_acc = [74.01, 74.07, 74.06, 74.05, 74.02, 74.01, 74.00, 73.99, 73.98, 73.96]

trace_c = go.Scatter(
    x=x_cb,
    y=lambda_c_acc,
    mode="lines+markers",
    name=r"$\Huge\lambda_\mathrm{c}$",
    line=dict(color="darkred", width=6),
    marker=dict(size=20),
)

trace_b = go.Scatter(
    x=x_cb,
    y=lambda_b_acc,
    mode="lines+markers",
    name=r"$\Huge\lambda_\mathrm{b}$",
    line=dict(color="olive", width=6),
    marker=dict(size=20),
)

layout_combined = go.Layout(
    xaxis=dict(
        title="Hyperparameter Value",
        title_standoff=10,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="lightgrey",
        gridwidth=1,
    ),
    yaxis=dict(
        title="Accuracy (%)",
        title_standoff=50,
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="lightgrey",
        gridwidth=1,
        range=[72.0, 75],
    ),
    legend=dict(
        x=0.5,
        y=0.01,
        font=dict(size=36, family="Times New Roman"),
    ),
    plot_bgcolor="white",
    font=dict(size=36, family="Times New Roman", color="black"),
    margin=dict(l=5, r=2, t=40, b=120),
    width=1700,
    height=650,
)

fig_combined = go.Figure(data=[trace_c, trace_b], layout=layout_combined)
fig_combined.write_image("lambda_cb_plot.pdf")
