import argparse
import collections
import torch
import os
import time

from utils import set_random_seed

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
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

import trainers.coop
import trainers.cocoop
import trainers.semobridge
import trainers.clip_adapter

import evaluators.classification_gpu

torch.serialization.add_safe_globals(
    [torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.StepLR]
)
torch.serialization.add_safe_globals([torch.optim.SGD])
torch.serialization.add_safe_globals([torch.optim.AdamW])

torch.serialization.add_safe_globals([collections.defaultdict])
torch.serialization.add_safe_globals([dict])


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.SEMOBRIDGE = CN()
    cfg.TRAINER.SEMOBRIDGE.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.SEMOBRIDGE.TEXTS = "cupl_full"
    cfg.TRAINER.SEMOBRIDGE.CSB = True  # class-bias

    cfg.TRAINER.CLIP_ADAPTER = CN()
    cfg.TRAINER.CLIP_ADAPTER.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.CLIP_ADAPTER.TEXTS = "cupl_full"

    cfg.DATASET.DIRECTORY = ""  # path to dataset
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.TRAIN.LOSSES = []

    cfg.TRAIN.IMAGE_LOSS_WEIGHT = 1.0  # weight for image loss
    cfg.TRAIN.TEXT_LOSS_WEIGHT = 1.0  # weight for text loss
    cfg.TRAIN.CONSISTENCY_LOSS_WEIGHT = 1.0  # weight for consistency loss
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT = 1.0

    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT = CN()  # 0.0 -> All text, 1.0 -> All image
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["OxfordPets"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["OxfordFlowers"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["FGVCAircraft"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["DescribableTextures"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["EuroSAT"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["StanfordCars"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["Food101"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["SUN397"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["Caltech101"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["UCF101"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["ImageNet"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["ImageNetSketch"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["ImageNetV2"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["ImageNetA"] = 0.5
    cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT["ImageNetR"] = 0.5

    cfg.TRAIN.CONS_LOSS_WEIGHT = CN()
    cfg.TRAIN.CONS_LOSS_WEIGHT["OxfordPets"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["OxfordFlowers"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["FGVCAircraft"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["DescribableTextures"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["EuroSAT"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["StanfordCars"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["Food101"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["SUN397"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["Caltech101"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["UCF101"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["ImageNet"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["ImageNetSketch"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["ImageNetV2"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["ImageNetA"] = 0.1
    cfg.TRAIN.CONS_LOSS_WEIGHT["ImageNetR"] = 0.1

    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT = CN()
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["OxfordPets"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["OxfordFlowers"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["FGVCAircraft"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["DescribableTextures"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["EuroSAT"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["StanfordCars"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["Food101"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["SUN397"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["Caltech101"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["UCF101"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["ImageNet"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["ImageNetSketch"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["ImageNetV2"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["ImageNetA"] = 0.1
    cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT["ImageNetR"] = 0.1

    cfg.TRAIN.RANDOMIZE_BATCHES = False
    cfg.TRAIN.LABEL_SMOOTHING = 0.0  # label smoothing

    cfg.TRAIN.TEXT_LENGTH_INIT = 27.5  # initial text length, from SD-IPC
    cfg.TRAIN.TEXT_LENGTH_TRAINABLE = False  # If text length is trainable

    cfg.INPUT.FEW_SHOT_AUGMENTATION = CN()
    cfg.INPUT.FEW_SHOT_AUGMENTATION["1"] = CN()
    cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].AUGMENT_EPOCHS = (
        0  # number of epochs to augment the shots
    )
    cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].AUGMENT_TRANSFORMS = [
        "random_resized_crop",
        "random_flip",
    ]  # data augmentation methods
    cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].RRCROP_SCALE = [0.8, 1.0]
    cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].COLORJITTER_B = 0.4  # brightness
    cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].COLORJITTER_C = 0.4  # contrast
    cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].COLORJITTER_S = 0.4  # saturation
    cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].COLORJITTER_H = 0.1  # hue
    # Copy the above for all number of shots
    cfg.INPUT.FEW_SHOT_AUGMENTATION["2"] = cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].clone()
    cfg.INPUT.FEW_SHOT_AUGMENTATION["4"] = cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].clone()
    cfg.INPUT.FEW_SHOT_AUGMENTATION["8"] = cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].clone()
    cfg.INPUT.FEW_SHOT_AUGMENTATION["16"] = cfg.INPUT.FEW_SHOT_AUGMENTATION["1"].clone()

    cfg.TRAIN.EARLY_STOPPING = CN()
    cfg.TRAIN.EARLY_STOPPING.ENABLED = False  # enable early stopping
    cfg.TRAIN.EARLY_STOPPING.PATIENCE = 0  # patience for early stopping
    cfg.TRAIN.EARLY_STOPPING.THRESHOLD = 0.0  # threshold for early stopping

    cfg.OPTIM.PATIENCE = 0  # patience for early stopping

    cfg.LOGITS = ["z1", "z2", "z3", "z4"]  # logits to use for evaluation
    cfg.TEST.EVALUATOR = "ClassificationGPU"
    cfg.TEST.FINAL_MODEL = "best_val"

    cfg.HP_SEARCH = CN()
    cfg.HP_SEARCH.ENABLED = True  # enable hyperparameter search
    cfg.HP_SEARCH.N_TRIALS = 1000
    cfg.HP_SEARCH.PARAMS = CN()

    # PARAMS is dictionary
    cfg.HP_SEARCH.PARAMS["smoothness"] = CN()
    cfg.HP_SEARCH.PARAMS["smoothness"].START = 0.0
    cfg.HP_SEARCH.PARAMS["smoothness"].MAX = 1.5
    cfg.HP_SEARCH.PARAMS["smoothness"].INIT = 0.01

    cfg.HP_SEARCH.PARAMS["alpha"] = CN()
    cfg.HP_SEARCH.PARAMS["alpha"].START = 0.0
    cfg.HP_SEARCH.PARAMS["alpha"].MAX = 10.0
    cfg.HP_SEARCH.PARAMS["alpha"].INIT = 1.0

    cfg.HP_SEARCH.PARAMS["beta"] = CN()
    cfg.HP_SEARCH.PARAMS["beta"].START = 0.0
    cfg.HP_SEARCH.PARAMS["beta"].MAX = 10.0
    cfg.HP_SEARCH.PARAMS["beta"].INIT = 1.0

    cfg.HP_SEARCH.PARAMS["gamma"] = CN()
    cfg.HP_SEARCH.PARAMS["gamma"].START = 0.0
    cfg.HP_SEARCH.PARAMS["gamma"].MAX = 10.0
    cfg.HP_SEARCH.PARAMS["gamma"].INIT = 1.0

    cfg.HP_SEARCH.PARAMS["delta"] = CN()
    cfg.HP_SEARCH.PARAMS["delta"].START = 0.0
    cfg.HP_SEARCH.PARAMS["delta"].MAX = 10.0
    cfg.HP_SEARCH.PARAMS["delta"].INIT = 1.0

    cfg.HP_SEARCH.PARAMS["lambda1"] = CN()
    cfg.HP_SEARCH.PARAMS["lambda1"].START = 0.0
    cfg.HP_SEARCH.PARAMS["lambda1"].MAX = 1.0
    cfg.HP_SEARCH.PARAMS["lambda1"].INIT = 0.5

    cfg.HP_SEARCH.PARAMS["lambda2"] = CN()
    cfg.HP_SEARCH.PARAMS["lambda2"].START = 0.0
    cfg.HP_SEARCH.PARAMS["lambda2"].MAX = 1.0
    cfg.HP_SEARCH.PARAMS["lambda2"].INIT = 0.5

    cfg.HP_SEARCH.PARAMS["lambda3"] = CN()
    cfg.HP_SEARCH.PARAMS["lambda3"].START = 0.0
    cfg.HP_SEARCH.PARAMS["lambda3"].MAX = 1.0
    cfg.HP_SEARCH.PARAMS["lambda3"].INIT = 0.5

    cfg.HP_SEARCH.PARAMS["lambda4"] = CN()
    cfg.HP_SEARCH.PARAMS["lambda4"].START = 0.0
    cfg.HP_SEARCH.PARAMS["lambda4"].MAX = 1.0
    cfg.HP_SEARCH.PARAMS["lambda4"].INIT = 0.1


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    print(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)

        trainer.test(vis=args.vis, eval_only=True)

        if args.diffusion:
            trainer.diffusion()
        return

    if not args.no_train:
        trainer.train(vis=args.vis)
    else:
        trainer.test(vis=args.vis, eval_only=True)

    if args.diffusion:
        trainer.diffusion()


def run_hp_search(args):
    # Create the output directory
    if not os.path.exists("OUTPUT/HP_SEARCH/"):
        os.makedirs("OUTPUT/HP_SEARCH/")

    # Create the log file
    search_log_file = os.path.join("OUTPUT/HP_SEARCH/", "SEARCH_RESULTS.txt")
    # Define the hyperparameter search space

    # Linear from 0.0 to 1.0
    image_text_loss_weights = torch.linspace(0.3, 1.0, 8)
    text_loss_weights = torch.linspace(0.0, 2.0, 22)
    consistency_loss_weights = torch.linspace(0.0, 2.0, 22)
    bias_norm_loss_weights = torch.linspace(0.0, 2.0, 22)
    # text_length_init = torch.linspace(1.0, 40.0, 39)

    for image_text_loss_weight in image_text_loss_weights:
        # for text_loss_weight in text_loss_weights:
        cfg = setup_cfg(args)
        cfg.defrost()
        cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT[cfg.DATASET.NAME] = (
            image_text_loss_weight.item()
        )
        # cfg.TRAIN.TEXT_LOSS_WEIGHT = text_loss_weight.item()
        # cfg.TRAIN.CONSISTENCY_LOSS_WEIGHT = consistency_loss_weight.item()
        # cfg.TRAIN.TEXT_LENGTH_INIT = text_length_init.item()

        run_name = "image_text_loss_weight_{}".format(image_text_loss_weight.item())

        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, run_name)
        cfg.freeze()

        if cfg.SEED >= 0:
            print("Setting fixed seed: {}".format(cfg.SEED))
            set_random_seed(cfg.SEED)
        setup_logger(cfg.OUTPUT_DIR)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            torch.backends.cudnn.benchmark = True

        print_args(args, cfg)
        print("Collecting env info ...")
        print("** System info **\n{}\n".format(collect_env_info()))

        trainer = build_trainer(cfg)

        print(cfg)

        accuracy = trainer.train()

        # Load the log file
        result = f"{run_name}, Accuracy: {accuracy:.2f}%"
        print(result)

        with open(search_log_file, "a") as f:
            f.write(result + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--diffusion", action="store_true", help="run diffusion process"
    )
    parser.add_argument("--vis", action="store_true", help="visualizations")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "--hp-search", action="store_true", help="run hyperparameter search"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()

    if not args.hp_search:
        main(args)
    else:
        run_hp_search(args)
