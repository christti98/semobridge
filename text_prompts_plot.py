import plotly.graph_objs as go

# X-axis: number of shots
x = [1, 2, 4, 8, 16]

# Accuracy values for each prompt type (placeholder/example)
aphotoofa_template = [71.96, 75.19, 77.62, 80.01, 82.42]
clip_prompts = [72.32, 75.30, 77.81, 80.07, 82.39]
clip_ensemble = [72.51, 75.45, 77.88, 80.09, 82.47]
cupl = [74.01, 76.28, 78.45, 80.53, 82.63]
clip_and_cupl = [74.02, 76.33, 78.55, 80.50, 82.66]

marker_size = 20
line_width = 4

# Define all traces
traces = [
    go.Scatter(
        x=x,
        y=aphotoofa_template,
        mode="lines+markers",
        name=r'"a photo of a {}"',
        line=dict(color="orange", width=line_width, dash="dash"),
        marker=dict(size=marker_size),
    ),
    go.Scatter(
        x=x,
        y=clip_prompts,
        mode="lines+markers",
        name="CLIP prompts",
        line=dict(color="green", width=line_width),
        marker=dict(size=marker_size),
    ),
    go.Scatter(
        x=x,
        y=clip_ensemble,
        mode="lines+markers",
        name="CLIP ensemble",
        line=dict(color="olive", width=line_width),
        marker=dict(size=marker_size),
    ),
    go.Scatter(
        x=x,
        y=cupl,
        mode="lines+markers",
        name="CuPL prompts",
        line=dict(color="darkred", width=line_width),
        marker=dict(size=marker_size),
    ),
    go.Scatter(
        x=x,
        y=clip_and_cupl,
        mode="lines+markers",
        name="CLIP ensemble + CuPL prompts",
        line=dict(color="purple", width=line_width),
        marker=dict(size=marker_size),
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
        font=dict(size=34, family="Times New Roman"),
        borderwidth=0,
    ),
    plot_bgcolor="white",
    font=dict(size=40, family="Times New Roman", color="black"),
    margin=dict(l=5, r=2, t=40, b=120),
)

# Create and save figure
fig = go.Figure(data=traces, layout=layout)
fig.write_image(
    "shots_vs_prompt_type_accuracy.pdf",
    width=1200,
    height=1000,
)
