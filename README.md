# SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP

This repo contains code needed to reproduce the results for SeMoBridge and SeMoBridge-T.

## How to Install

### Dassl
This code is built on top of the code for [CoOp](https://github.com/KaiyangZhou/CoOp). It uses the toolbox Dassl.pytorch, so you need to install the `dassl` environment first. ONLY USE THE INCLUDED DASSL, as it contains a fix to work with PyTorch 2.7.0.

Follow this to install the dassl conda environment:
```
cd Dassl.pytorch/

# Create a conda environment
conda create -y -n dassl python=3.12.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 2.7.0) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, run `pip install -r requirements.txt` under `SeMoBridge/` to install a few more packages required (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to Run

We include four bash (`.sh`) scripts to make running SeMoBridge or SeMoBridge-T easier.

### Training-free SeMoBridge
`scripts/semobridge/run_all_datasets_notrain.sh` will run SeMoBridge on the standard 11 datasets for all shots (1, 2, 4, 8, 16) on three seeds (1, 2, 3).<br>
You may have to change the DATA path in the script to point to the datasets. 

To run SeMoBridge ViT-B/16 on everything, use the command<br>`bash scripts/semobridge/run_all_datasets_notrain.sh vit_b16 clip_ensemble,cupl_full True`.
<br>The results will be saved in a .csv file. For example `OUTPUT/SeMoBridge_vit_b16_submit3_clip_ensemble,cupl_full_cbTrue.csv`.

### Training SeMoBridge-T
SeMoBridge-T training is done in the same way:<br>`bash scripts/semobridge/run_all_datasets.sh vit_b16 clip_ensemble,cupl_full True`

The scripts have the following arguments:<br>`run_all_datasets_notrain.sh [CONFIG FILE] [TEXT PROMPTS] [CSB False/True] [DATASET CONFIG NAME] [NUMBER OF SHOTS]`.<br>

For example, to run SeMoBridge-T on 16-shot ImageNet with ViT-B/16 without CSB, use<br>`bash scripts/semobridge/run_all_datasets.sh vit_b16 clip_ensemble,cupl_full False imagenet 16`

### OOD
To run out of distribution (OOD) datasets, you can use `scripts/semobridge/ood_notrain.sh` and `scripts/semobridge/ood.sh`.<br>
They support the arguments<br>`ood_notrain.sh [CONFIG FILE] [TEXT PROMPTS] [CSB False/True] [NUMBER OF SHOTS]`

To run SeMoBridge-T on OOD datasets, it has to be trained on ImageNet first, as the OOD script will load the model from the OUTPUT directory.

### Visualizations and Plots
To generate Figures 2 and 4, manually run `train.py` with the `--vis` argument.<br>
Other figures are generated with `bias_norms_plot.py`, `draw_curves.py` (from a custom csv file), `lambda_plot.py`, and `text_prompts_plot.py`.

### Notes
Preprocessed features will be saved in `preprocessed/`. They may have to be deleted if changes to the code are made.

## Acknowledgements
We based our implementation on several excellent repositories: [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp),  [TIP-Adapter](https://github.com/gaopengcuhk/Tip-Adapter/), [SuS-X](https://github.com/vishaal27/SuS-X), [LDC](https://github.com/LiShuo1001/LDC), [APE](https://github.com/yangyangyang127/APE), and [CuPL](https://github.com/sarahpratt/CuPL). We thank the respective authors for providing open access to their work.