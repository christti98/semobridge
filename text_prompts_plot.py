import plotly.graph_objs as go

# X-axis: number of shots
x = [1, 2, 4, 8, 16]

# Accuracy values for each prompt type (placeholder/example)
aphotoofa_template = [71.73, 74.96, 77.42, 79.56, 81.90] # NEW
clip_prompts = [72.10, 75.11, 77.42, 79.71, 81.83] # NEW
clip_ensemble = [72.31, 75.23, 77.63, 79.73, 82.01] # NEW
cupl = [73.88, 76.08, 78.20, 80.07, 82.19] # NEW
clip_and_cupl = [74.01, 76.20, 78.21, 80.10, 82.20] # NEW

colors = ["#B49250", "#F9A800", "#F97400", "#D35FDB", "#87218F"]

symbols = ["circle", "square", "diamond", "cross", "star"]


# Make color into rgba with alpha
for i, color in enumerate(colors):
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        colors[i] = f"rgba({r},{g},{b},0.75)"

marker_size = 20
line_width = 6

# Define all traces
traces = [
    go.Scatter(
        x=x,
        y=aphotoofa_template,
        mode="lines+markers",
        name=r'"a photo of a {}"',
        line=dict(color=colors[0], width=line_width, dash="dash"),
        marker=dict(size=marker_size, symbol=symbols[0]),
    ),
    go.Scatter(
        x=x,
        y=clip_prompts,
        mode="lines+markers",
        name="CLIP prompts",
        line=dict(color=colors[1], width=line_width),
        marker=dict(size=marker_size, symbol=symbols[1]),
    ),
    go.Scatter(
        x=x,
        y=clip_ensemble,
        mode="lines+markers",
        name="CLIP ensemble",
        line=dict(color=colors[2], width=line_width),
        marker=dict(size=marker_size, symbol=symbols[2]),
    ),
    go.Scatter(
        x=x,
        y=cupl,
        mode="lines+markers",
        name="CuPL",
        line=dict(color=colors[3], width=line_width),
        marker=dict(size=marker_size, symbol=symbols[3]),
    ),
    go.Scatter(
        x=x,
        y=clip_and_cupl,
        mode="lines+markers",
        name="CLIP ensemble+CuPL",
        line=dict(color=colors[4], width=line_width),
        marker=dict(size=marker_size, symbol=symbols[4]),
    ),
]

# Layout
layout = go.Layout(
    xaxis=dict(
        title="Number of Shots",
        type="log",
        tickvals=[1, 2, 4, 8, 16],
        ticktext=["1", "2", "4", "8", "16"],
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        gridcolor="lightgrey",
        gridwidth=1,
        title_standoff=10,
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
    ),
    legend=dict(
        x=0.99,  # just outside the right edge
        y=0.01,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=26, family="Times New Roman"),
        borderwidth=0,
    ),
    plot_bgcolor="white",
    font=dict(size=36, family="Times New Roman", color="black"),
    margin=dict(l=5, r=2, t=5, b=120),
)

# Create and save figure
fig = go.Figure(data=traces, layout=layout)
fig_path = "shots_vs_prompt_type_accuracy.pdf"
fig.write_image(
    fig_path.replace(".pdf", ".pdf"),
    width=800,
    height=500,
)

fig.write_image(
    fig_path.replace(".pdf", ".svg"),
    width=800,
    height=500,
)