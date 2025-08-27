# Load SeMoBridge

import torch
from trainers.semobridge import SeMoBridge

import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

from dassl.data import DataManager

from train import get_cfg_default, extend_cfg

# Make fake cfg to load the model
from yacs.config import CfgNode as CN

dataset_config_file = f"configs/datasets/ucf101.yaml"
method_config_file = f"configs/trainers/SeMoBridge/vit_b16.yaml"

cfg = get_cfg_default()
extend_cfg(cfg)

cfg.DATASET.ROOT = "DATA"
cfg.DATASET.NUM_SHOTS = 16

# Load dataset config
cfg.merge_from_file(dataset_config_file)
cfg.merge_from_file(method_config_file)

cfg.INPUT.FEW_SHOT_AUGMENTATION["16"].AUGMENT_EPOCHS = 0

# Load dataset to get classnames
dm = DataManager(cfg)

# Load classnames from file
# classnames_path = "DATA/fgvc_aircraft/variants.txt"
# with open(classnames_path, "r") as f:
#     classnames = [line.strip() for line in f.readlines()]

num_classes = dm.dataset.num_classes
classnames = dm.dataset.classnames

# If classnames are too long, truncate them
for i in range(len(classnames)):
    if len(classnames[i]) > 12:
        classnames[i] = classnames[i][:12] + "..."

dataset_dir = cfg.DATASET.DIRECTORY.replace("-", "").lower()

if cfg.DATASET.NAME == "DescribableTextures":
    cfg.DATASET.NAME = "DTD"

# Load the model with L_bias
model_path = f"OUTPUT/{dataset_dir}/SeMoBridge/vit_b16_cbTrue/16shots/texts_clip_ensemble,cupl_semobridge/seed1/semobridge/model.pth.tar-5000"


# Fake text projection
text_projection = torch.zeros(512, 512)

# Fake text embedding length
avg_text_embedding_length_classwise = torch.ones(num_classes)
avg_text_embedding_length = torch.tensor(1.0)
dtype = torch.float32

semobridge = SeMoBridge(
    cfg,
    text_projection,
    avg_text_embedding_length_classwise,
    avg_text_embedding_length,
    num_classes,
    dtype,
)
semobridge.load_state_dict(
    torch.load(model_path, map_location="cpu", weights_only=False)["state_dict"]
)
semobridge.eval()
bias_norms_with = semobridge.class_bias.norm(dim=-1).detach().cpu().numpy()

# Load the model without L_bias
model_path_no_lbias = f"OUTPUT/{dataset_dir}/SeMoBridge/vit_b16_img_txtp_txte_cons_cbTrue/16shots/texts_clip_ensemble,cupl_semobridge/seed1/semobridge/model.pth.tar-5000"
semobridge = SeMoBridge(
    cfg,
    text_projection,
    avg_text_embedding_length_classwise,
    avg_text_embedding_length,
    num_classes,
    dtype,
)
semobridge.load_state_dict(
    torch.load(model_path_no_lbias, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
)
semobridge.eval()
bias_norms_without = semobridge.class_bias.norm(dim=-1).detach().cpu().numpy()

## ========================= PLOT BIAS NORMS COMPARISON =========================
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Generate angles
angles_deg = np.linspace(0, 360, num_classes, endpoint=False)
angles_deg = np.append(angles_deg, angles_deg[0])
bias_norms_with = np.append(bias_norms_with, bias_norms_with[0])
bias_norms_without = np.append(bias_norms_without, bias_norms_without[0])

# Class tick positions (e.g., every ~30°)
num_ticks = 16
tick_indices = np.linspace(
    0, int(num_classes - (num_classes / num_ticks)), num_ticks, dtype=int
)
tick_angles = tick_indices * (360.0 / num_classes)
tick_labels = [f"{classnames[i]}" for i in tick_indices]

radial_range = (0, max(np.max(bias_norms_with), np.max(bias_norms_without)) + 0.5)
radial_tick_vals = list(range(0, int(radial_range[1]) + 1))

# Create subplot layout
fig = go.Figure()

# Plot: With L_bias
fig.add_trace(
    go.Scatterpolar(
        r=bias_norms_with,
        theta=angles_deg,
        mode="lines",
        name=r"$\large\text{with}~\mathcal{L}_{\mathrm{bias}}$",
        line=dict(color="green", width=3, dash="dash"),
    )
)

# Plot: Without L_bias
fig.add_trace(
    go.Scatterpolar(
        r=bias_norms_without,
        theta=angles_deg,
        mode="lines",
        name=r"$\large\text{without}~\mathcal{L}_{\mathrm{bias}}$",
        line=dict(color="darkred", width=2.5),
    )
)

# Layout update
fig.update_layout(
    title=dict(
        text=f"{cfg.DATASET.NAME} CSB Norms",
        x=0.01,
        xanchor="left",
        yanchor="top",
        y=0.97,
        font=dict(size=22, family="Times New Roman", weight=500, color="black"),
    ),
    font=dict(size=20, family="Times New Roman"),
    margin=dict(l=100, r=100, t=5, b=5),
    showlegend=True,
    legend=dict(
        font=dict(size=16),
        yanchor="top",
        y=1.01,
        xanchor="right",
        x=1.25,
        orientation="v",
    ),
    paper_bgcolor="white",  # Background outside the polar plot
    plot_bgcolor="white",  # Background inside the polar plot
    polar=dict(
        bgcolor="white",  # background inside polar plot
        domain=dict(x=[0, 1], y=[0, 1]),
        radialaxis=dict(
            visible=True,
            showline=True,
            linewidth=2,
            linecolor="black",
            gridcolor="lightgrey",
            gridwidth=2,
            tickfont=dict(size=14, weight=600),
            tickangle=0,
            ticks="outside",
            ticklen=6,
            tickwidth=2,
            tickmode="array",
            tickvals=radial_tick_vals,
            ticktext=[str(v) for v in radial_tick_vals],
        ),
        angularaxis=dict(
            tickmode="array",
            tickvals=tick_angles,
            ticktext=tick_labels,
            tickfont=dict(size=14, weight=600),
            rotation=90,
            direction="clockwise",
            ticks="outside",
            ticklen=6,
            tickwidth=2,
            linewidth=2,
            linecolor="grey",
            gridcolor="lightgrey",  # <––– change color of angular grid lines here
            gridwidth=1,  # optional: make them more visible
        ),
    ),
)

# Annotation for "Bias Norm" (left polar plot)
# fig.add_annotation(
#     text=r"$\large\text{with}~\mathcal{L}_{\mathrm{bias}}$",
#     xref="paper",
#     yref="paper",
#     x=0.50,
#     y=0.76,  # slightly above middle
#     xanchor="center",
#     yanchor="bottom",
#     showarrow=False,
#     font=dict(size=22, family="Times New Roman"),
# )

# # Annotation for "Bias Norm" (right polar plot)
# fig.add_annotation(
#     text=r"$\large\text{without}~\mathcal{L}_{\mathrm{bias}}$",
#     xref="paper",
#     yref="paper",
#     x=0.50,
#     y=0.26,
#     xanchor="center",
#     yanchor="bottom",
#     showarrow=False,
#     font=dict(size=22, family="Times New Roman"),
# )

fig.add_annotation(
    text=r"$\lVert\hat{\mathbf{b}}\rVert$",
    font=dict(size=20, family="Times New Roman"),
    showarrow=False,
    xref="paper",
    yref="paper",
    x=0.40,  # Left side of plot
    y=0.505,  # Vertically centered
    textangle=0,
)


# Save figure
fig.write_image(f"bias_norms/{dataset_dir}.pdf", width=500, height=500)
