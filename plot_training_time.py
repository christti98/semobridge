import plotly.graph_objects as go

# === DATA & CONFIGURATION (from the first plot) ===
# This data is extracted from your first image.
methods = {
    'CLIP': {'time': 0, 'acc': 65.52, 'symbol': 'triangle-up', 'color': '#C71585', 'size': 18},
    'APE': {'time': 0, 'acc': 75.328, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "ICCV'23", "y_shift": -10},
    'SeMoBridge': {'time': 0, 'acc': 75.404, 'symbol': 'star', 'color': 'purple', 'size': 18, "venue": "Ours", "y_shift": 10},

    #'Tip-X': {'time': 0.01, 'acc': 62.1, 'symbol': 'star', 'color': '#DAA520', 'size': 18},
    'Tip-Adapter': {'time': 0, 'acc': 73.434, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "ECCV'21"},
    'APE-T': {'time': 60*3.5, 'acc': 77.176, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "ICCV'23"},
    'CLIP-LoRA': {'time': 759, 'acc': 77.634, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "CVPR'24", "x_anchor": "right"},
    'CLIP-Adapter': {'time': 60*32, 'acc': 69.448, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "IJCV'21"},
    'Tip-Adapter-F': {'time': 3.5*60, 'acc': 75.896, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "ECCV'21"},
    #'CoOp': {'time': 100, 'acc': 63.9, 'symbol': 'circle', 'color': '#4169E1', 'size': 18},
    #'CoCoOp': {'time': 100, 'acc': 63.2, 'symbol': 'circle', 'color': '#4169E1', 'size': 18},
    'LDC': {'time': 60*1+54, 'acc': 77.166, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "CVPR'25", "x_anchor": "left", "x_shift": -25, "y_shift": -25},
    'PromptSRC': {'time': 6153.6, 'acc': 77.904, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "ICCV'23"},
    '2SFS': {'time': 4*60+29, 'acc': 77.79, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "CVPR'25", "y_shift": 8},
    'SkipT': {'time': 405.8, 'acc': 77.428, 'symbol': 'circle', 'color': '#DC143C', 'size': 18, "venue": "CVPR'25"},

    'SeMoBridge-T': {'time': 27, 'acc': 78.144, 'symbol': 'star', 'color': 'purple', 'size': 18, "venue": "Ours"},
}

# === PLOT CREATION ===
fig = go.Figure()

# Add the scatter plot trace
fig.add_trace(go.Scatter(
    x=[d['time'] for d in methods.values()],
    y=[d['acc'] for d in methods.values()],
    mode='markers',
    marker=dict(
        symbol=[d['symbol'] for d in methods.values()],
        color=[d['color'] for d in methods.values()],
        size=[d['size'] for d in methods.values()],
    ),
    showlegend=False
))

# Add annotations (text labels) for each point
annotations = []

for name, data in methods.items():
    x_anchor = data.get("x_anchor", "left")
    font_style = {'family': "Times New Roman", 'color': 'black', 'size': 24, "weight": "normal"}
    x_shift = data.get("x_shift", 10 if x_anchor=='left' else -10)

    if "SeMoBridge" in name:
        font_style["color"] = 'purple'
        font_style["size"] = 28
        font_style["weight"] = 'bold'

    annotations.append(dict(
        x=data['time'], y=data['acc'], 
        text=f"{name} <span style='color:{"purple" if "SeMoBridge" in name else "grey"}'>({data['venue']})</span>" if "venue" in data else name, 
        showarrow=False,
        xref="x", yref="y", xanchor=x_anchor, yanchor='middle',
        yshift=data.get("y_shift", 0), xshift=x_shift,
        font=font_style
    ))

grid_color = "rgb(230, 230, 230)"

# === LAYOUT STYLING (from the second plot script) ===
fig.update_layout(
    # title=dict(
    #     text="Model Performance vs. Training Time",
    #     x=0.5,
    #     xanchor="center"
    # ),
    # --- CHANGE 2 & 3: Update x-axis to linear scale in seconds with minute ticks ---
    xaxis=dict(
        title="Training Time (min)",
        type='linear', # Change from 'log' to 'linear'
        range=[-10, 770], # Set a new linear range
        tickvals=[i * 60 for i in range(17)], # Ticks every 60 seconds
        ticktext=[f"{i}" for i in range(17)], # Label ticks with minute markers
        showgrid=False,
        tickcolor="black",
        ticks="outside",
        tickwidth=2,

        # light grey grid lines
        gridcolor=grid_color,
        gridwidth=1,
        # add grid line on 0
        zeroline=False,
        zerolinecolor=grid_color,
    ),
    # --- CHANGE 4: Update y-axis range to fit new data ---
    yaxis=dict(
        title="Accuracy (%)",
        range=[75, 78.3], # Adjust range for new accuracy values
        showgrid=False,

        tickcolor="black",
        ticks="outside",
        tickwidth=2,
        # Tick every 0.5 percent
        dtick=0.5,

        # light grey grid lines
        gridcolor=grid_color,
        gridwidth=1,
    ),
    font=dict(size=26, family="Times New Roman", weight="normal", color="black"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=50, r=5, t=5, b=50),
    annotations=annotations
)

# Add border lines around the entire plot area
fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=False)
fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=False)

# To save the figure, you can use:
fig.write_image("accuracy_vs_training_time.pdf", width=700, height=450)
#fig.show()