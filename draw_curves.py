import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.colors import qualitative

# === CONFIGURATION ===
csv_path = "Accuracy SeMoBridge - RN-50 FINAL.csv"
save_dir = "main_curves"
os.makedirs(save_dir, exist_ok=True)

# === LOAD & CLEAN DATA ===
raw = pd.read_csv(csv_path, skiprows=1)
header_row = pd.read_csv(csv_path, nrows=1).iloc[0]

first_row = pd.read_csv(csv_path, nrows=2).iloc[1]

model = first_row.iloc[0]

# Rename columns
new_columns = ["Model", "Method", "Shots"]
new_columns += header_row[3:-2].dropna().tolist()
new_columns += ["AVERAGE ACCURACY", "AVERAGE TRAINING TIME"]
raw.columns = new_columns
raw["Shots"] = pd.to_numeric(raw["Shots"], errors="coerce")

# === SETUP ===
datasets = [
    "OxfordPets",
    "Flowers102",
    "FGVCAircraft",
    "DTD",
    "EuroSAT",
    "StanfordCars",
    "Food101",
    "SUN397",
    "Caltech101",
    "UCF101",
    "ImageNet",
]

shot_ticks = [0, 1, 2, 4, 8, 16]
tick_map = {0: 0, 1: 0.2, 2: 1.2, 4: 2.2, 8: 3.2, 16: 4.2}

# Define methods as a dictionary with their attributes (name, type, color, symbol)
methods_dict = {
    # TRAINING-FREE METHODS
    "CLIP zero-shot": {"type": "training-free", "color": "#6CC557", "symbol": "cross"},
    "Tip-Adapter": {"type": "training-free", "color": "#F9A800", "symbol": "circle"},
    "Tip-X": {"type": "training-free", "color": "#FF6600", "symbol": "circle"},
    "APE": {"type": "training-free", "color": "#4E95D9", "symbol": "circle"},
    "SeMoBridge": {"type": "training-free", "color": "#87218F", "symbol": "star"},
    # TRAINING METHODS
    "CoOp": {"type": "training", "color": "#6CC557", "symbol": "circle"},
    "CLIP-Adapter": {"type": "training", "color": "#B49250", "symbol": "circle"},
    "Tip-Adapter-F": {"type": "training", "color": "#F9A800", "symbol": "circle"},
    "APE-T": {"type": "training", "color": "#4E95D9", "symbol": "circle"},
    "LDC": {"type": "training", "color": "#38DFAA", "symbol": "circle"},
    "SeMoBridge-T": {"type": "training", "color": "#87218F", "symbol": "star"},
}


# Helper to extract numeric accuracy
def extract_score(val):
    if pd.isna(val):
        return np.nan
    match = re.match(r"([\d.]+)", str(val))
    return float(match.group(1)) if match else np.nan


# === NEW METHOD TO GENERATE PLOT FOR SPECIFIC ARCHITECTURE ===
def plot_for_architecture_and_method(
    architecture: str, dataset: str, is_training_free: bool, average=False
):
    fig = go.Figure()
    ymin, ymax = float("inf"), float("-inf")

    # Filter methods based on architecture and training type
    methods_filtered = [
        m
        for m, properties in methods_dict.items()
        if (is_training_free and properties["type"] == "training-free")
        or (not is_training_free and properties["type"] == "training")
    ]

    if average:
        # Calculate the average accuracy for each shot across all datasets
        for method in methods_filtered:
            avg_scores = {shot: [] for shot in shot_ticks}

            for dataset in datasets:
                df = raw[raw["Method"] == method].copy()
                df["Score"] = df[dataset].apply(extract_score)

                x = df["Shots"].to_numpy(dtype=float)
                y = df["Score"].to_numpy(dtype=float)
                valid = ~np.isnan(x) & ~np.isnan(y)

                if not np.any(valid):
                    continue

                x, y = x[valid], y[valid]

                # Append the valid accuracy for each shot to the respective shot value
                for i, shot in enumerate(x):
                    if shot in avg_scores:
                        avg_scores[shot].append(y[i])

            # Now calculate the average accuracy for each shot value across datasets
            avg_y = []
            for shot in shot_ticks:
                shot_scores = avg_scores[shot]
                if len(shot_scores) > 0:
                    avg_y.append(
                        np.nanmean(shot_scores)
                    )  # Calculate mean while ignoring NaNs
                else:
                    avg_y.append(np.nan)  # If no valid scores, append NaN

            avg_y = np.array(avg_y)

            # Only plot if there's at least one valid average value
            if np.all(np.isnan(avg_y)):
                continue  # Skip if all values are NaN

            # Replace with pseudo-logarithmic mapping
            avg_x = np.array([tick_map.get(int(shot), np.nan) for shot in shot_ticks])

            fig.add_trace(
                go.Scatter(
                    x=avg_x,
                    y=avg_y,
                    mode="lines+markers" if len(avg_x) > 1 else "markers",
                    name=f"{method}",
                    marker=dict(
                        color=methods_dict[method]["color"],
                        size=18,
                        symbol=methods_dict[method]["symbol"],
                    ),
                    line=dict(color=methods_dict[method]["color"], width=4),
                )
            )
    else:
        for method in methods_filtered:
            df = raw[raw["Method"] == method].copy()
            df["Score"] = df[dataset].apply(extract_score)

            x = df["Shots"].to_numpy(dtype=float)
            y = df["Score"].to_numpy(dtype=float)
            valid = ~np.isnan(x) & ~np.isnan(y)

            if not np.any(valid):
                continue

            x, y = x[valid], y[valid]
            ymin = min(ymin, min(y))
            ymax = max(ymax, max(y))

            # Get method properties (color and symbol) from the dictionary
            symbol = methods_dict[method]["symbol"]
            color = methods_dict[method]["color"]

            # Replace with pseudo-logarithmic mapping
            x = np.array([tick_map.get(int(shot), np.nan) for shot in x[valid]])

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers" if len(x) > 1 else "markers",
                    name=method,
                    marker=dict(
                        color=color,
                        size=18,
                        symbol=symbol,
                    ),
                    line=dict(color=color, width=4),
                )
            )

    if ymin < ymax:
        diff = ymax - ymin
        y_range = [ymin - 0.05 * diff, ymax + 0.05 * diff]
    else:
        y_range = None

    x_range = [-0.1, 4.5]  # Set x-axis range to cover all shot ticks

    title = ""
    if average:
        title = f"{'Training-free' if is_training_free else 'Training'} Average of 11 Datasets"
    else:
        title = f"{'Training-free' if is_training_free else 'Training'} on {dataset}"

    tickvals = list(tick_map.values())
    ticktext = list(tick_map.keys())

    # If it is training, remove the 0 shot tick
    if not is_training_free:
        tickvals = tickvals[1:]
        ticktext = ticktext[1:]

    fig.update_layout(
        title=title,
        title_x=0.5,  # Center the title
        title_xanchor="center",  # Ensure that the title is anchored in the center
        xaxis=dict(
            title="Number of Shots",
            range=x_range,
            tickvals=tickvals,  # Use the tick values from the mapping
            ticktext=ticktext,  # Use the original shot values as labels
            showgrid=True,  # Enable grid lines for x-axis
            gridcolor="lightgray",  # Set grid line color (optional)
            gridwidth=1,  # Set grid line width (optional)
            tickcolor="black",  # Set the color of the ticks to black
            ticks="outside",  # Place the ticks outside the axis
            tickwidth=2,  # Set the width of the ticks (optional)
            type="linear",  # Use linear scale for x-axis
        ),
        yaxis=dict(
            title="Accuracy (%)",
            range=y_range,
            showgrid=True,  # Enable grid lines for y-axis
            gridcolor="lightgray",  # Set grid line color (optional)
            gridwidth=1,  # Set grid line width (optional)
            tickcolor="black",  # Set the color of the ticks to black
            ticks="outside",  # Place the ticks outside the axis
            tickwidth=2,  # Set the width of the ticks (optional)
        ),
        legend=dict(
            x=0.99,
            y=0.01,
            xanchor="right",
            yanchor="bottom",
            traceorder="reversed",  # Invert the order of the methods in the legend
        ),
        margin=dict(l=50, r=15, t=75, b=40),
        font=dict(size=30, family="Times New Roman"),
        # White background
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    # Add border lines around plot
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

    directory = os.path.join(
        save_dir,
        architecture.replace("/", ""),
        "training-free" if is_training_free else "training",
    )
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = f"{directory}/{dataset if not average else 'AVERAGE'}.pdf"
    fig.write_image(
        save_path,
        width=700,
        height=600,
    )
    print(f"Saved plot to {save_path}")


# For training-free, plot the average across all datasets and then individual datasets
architectures = ["RN-50"]

for architecture in architectures:
    # Plot average across all datasets
    plot_for_architecture_and_method(
        architecture, None, is_training_free=False, average=True
    )
    plot_for_architecture_and_method(
        architecture, None, is_training_free=True, average=True
    )

    # Plot individual datasets
    for dataset in datasets:
        plot_for_architecture_and_method(
            architecture, dataset, is_training_free=False, average=False
        )
        plot_for_architecture_and_method(
            architecture, dataset, is_training_free=True, average=False
        )
