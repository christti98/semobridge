import plotly.graph_objs as go

# === Combined plot for lambda_it, lambda_c, and lambda_b ===
x = [0.1, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
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
]
lambda_c_acc = [73.90, 73.91, 73.92, 73.99, 73.92, 74.02, 74.00, 73.98, 73.97]
lambda_b_acc = [74.07, 74.06, 74.05, 74.02, 74.01, 74.00, 73.99, 73.98, 73.98]

trace_it = go.Scatter(
    x=x,
    y=lambda_it_acc,
    mode="lines+markers",
    name=r"$\Huge\lambda_\mathrm{it}$",
    line=dict(color="rgba(249, 168, 0, 0.8)", width=6),
    marker=dict(size=20),
)
trace_c = go.Scatter(
    x=x,
    y=lambda_c_acc,
    mode="lines+markers",
    name=r"$\Huge\lambda_\mathrm{c}$",
    line=dict(color="rgba(78, 149, 217, 0.8)", width=6),
    marker=dict(size=20),
)

trace_b = go.Scatter(
    x=x,
    y=lambda_b_acc,
    mode="lines+markers",
    name=r"$\Huge\lambda_\mathrm{b}$",
    line=dict(color="rgba(56, 223, 170, 0.8)", width=6),
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
        orientation="h",
        xanchor="center",
        yanchor="bottom",
        x=0.5,
        y=0.01,
        font=dict(size=36, family="Times New Roman"),
        # Space between legend items
        itemwidth=50,
        itemsizing="constant",
    ),
    plot_bgcolor="white",
    font=dict(size=36, family="Times New Roman", color="black"),
    margin=dict(l=5, r=2, t=5, b=120),
)

fig_combined = go.Figure(data=[trace_it, trace_c, trace_b], layout=layout_combined)
fig_combined.write_image("lambda_it_c_b_plot.pdf", width=800, height=500)
