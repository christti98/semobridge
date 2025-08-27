import random
import numpy as np
import torch


def generate_train_split_slug(cfg) -> str:
    num_shots = cfg.DATASET.NUM_SHOTS
    augment_cfg = cfg.INPUT.FEW_SHOT_AUGMENTATION.get(str(num_shots), None)
    if augment_cfg is None:
        raise ValueError(
            f"Few-shot augmentation configuration for {num_shots} shots not found in cfg.INPUT.FEW_SHOT_AUGMENTATION."
        )

    if augment_cfg.AUGMENT_EPOCHS == 0:
        return f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}"
    else:
        name = f"shot_{cfg.DATASET.NUM_SHOTS}-seed_{cfg.SEED}-augment_{augment_cfg.AUGMENT_EPOCHS}-transforms_{augment_cfg.AUGMENT_TRANSFORMS}"
        if "random_resized_crop" in augment_cfg.AUGMENT_TRANSFORMS:
            name += f"-rrcropscale_{augment_cfg.RRCROP_SCALE}"
        if "color_jitter" in augment_cfg.AUGMENT_TRANSFORMS:
            name += f"-colorjitter_b_{augment_cfg.COLORJITTER_B}-colorjitter_c_{augment_cfg.COLORJITTER_C}-colorjitter_s_{augment_cfg.COLORJITTER_S}-colorjitter_h_{augment_cfg.COLORJITTER_H}"
        return name


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
