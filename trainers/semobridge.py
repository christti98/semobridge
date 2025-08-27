import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
import time
import datetime

from dassl.engine import TRAINER_REGISTRY, SimpleTrainer
from dassl.utils import (
    load_pretrained_weights,
    load_checkpoint,
    MetricMeter,
    AverageMeter,
    mkdir_if_missing,
)
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.data_manager import build_transform

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import os
from utils import set_random_seed


import plotly.graph_objects as go

from extract_features import FeatureExtractor

import optuna
from optuna.samplers import TPESampler


from plotly.subplots import make_subplots
from PIL import Image

from torchvision.transforms import CenterCrop

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


OOD_DATASETS = ["ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2"]


class SeMoBridgeModel(nn.Module):
    def __init__(
        self,
        cfg,
        text_projection,
        avg_text_embedding_length_classwise,
        avg_text_embedding_length,
        num_classes,
        dtype,
    ):
        super().__init__()
        self.cfg = cfg

        self.avg_text_embedding_length_classwise = (
            avg_text_embedding_length_classwise.view(-1, 1, 1)
        )
        self.avg_text_embedding_length = avg_text_embedding_length
        print(
            f"Using avg_text_embedding_length init: {self.avg_text_embedding_length.item()}"
        )
        if cfg.TRAIN.TEXT_LENGTH_TRAINABLE:
            self.avg_text_embedding_length = torch.nn.Parameter(
                self.avg_text_embedding_length
            )
            print("Using trainable avg_text_embedding_length")

        self.text_projection = text_projection
        self.inv_text = torch.linalg.pinv(text_projection, atol=0.3)
        self.inv_text_untrained = self.inv_text.clone()

        self.inv_text = torch.nn.Parameter(self.inv_text)

        if cfg.TRAINER.SEMOBRIDGE.CSB:
            print("Using class bias")
            self.class_bias = torch.nn.Parameter(
                torch.zeros(num_classes, text_projection.shape[0], dtype=dtype)
            )

    def forward(self, image_embeds, use_class_bias=True, use_untrained=False):
        # image_embeds shape: [num_classes, num_shots, projection_dims]
        # self.inv_text shape: [projection_dims, projection_dims]

        # self.inv is learnable
        # self.class_bias is learnable

        if not use_untrained:
            image_emb_proj = torch.matmul(image_embeds, self.inv_text)  # [C, K, D]
            # image_emb_proj = self.inv_text(image_embeds)  # [C, K, D]
        else:
            image_emb_proj = torch.matmul(image_embeds, self.inv_text_untrained)

        if self.cfg.TRAINER.SEMOBRIDGE.CSB and use_class_bias and not use_untrained:
            # Vectorized bias addition: class_bias [C, D] → broadcast to [C, K, D]
            image_emb_proj = image_emb_proj + self.class_bias.unsqueeze(
                1
            )  # [C, 1, D] + [C, K, D]

        # Normalize
        image_emb_proj = image_emb_proj / image_emb_proj.norm(dim=-1, keepdim=True)

        image_emb_proj = self.avg_text_embedding_length * image_emb_proj

        # Project the converted image embeddings into shared space
        final_prompts_projected = image_emb_proj @ self.text_projection

        return final_prompts_projected, image_emb_proj


@TRAINER_REGISTRY.register()
class SeMoBridge(SimpleTrainer):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.SEMOBRIDGE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        self.is_converted = False

        cfg = self.cfg
        # Run feature extractor
        self.features = FeatureExtractor(self.cfg, self.dm, self.device)
        self.features.extract_features()

        # Calculate average text embedding length from encoded texts
        self.avg_text_embedding_length_classwise = self.features.text_encoded.norm(
            dim=-1
        )
        self.avg_text_embedding_length = self.avg_text_embedding_length_classwise.mean()

        # Build SeMoBridge
        num_classes = self.features.num_classes
        if self.cfg.DATASET.NAME in OOD_DATASETS:
            # For OOD datasets, we use the number of classes from ImageNet to build SeMoBridge first, because we should load bridge trained on ImageNet
            num_classes = len(self.features.imagenet_classnames)
        self.semobridge = SeMoBridgeModel(
            self.cfg,
            self.features.text_projection,
            self.avg_text_embedding_length_classwise,
            self.avg_text_embedding_length,
            num_classes,
            self.features.dtype,
        )

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.semobridge, cfg.MODEL.INIT_WEIGHTS)

        self.semobridge.to(self.device)

        # Set SeMoBridge to training mode
        self.semobridge.train()
        # Set parameters to require gradients
        for param in self.semobridge.parameters():
            param.requires_grad = True

        # NOTE: only give SeMoBridge to the optimizer
        self.optim = build_optimizer(self.semobridge, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("semobridge", self.semobridge, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.SEMOBRIDGE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.semobridge = nn.DataParallel(self.semobridge)

        print(self.get_model_names())

    def convert_final_prompts(self, image_embeds):
        # Convert the image embeddings to text embeddings
        converted_projected, converted_unprojected = self.semobridge(image_embeds)

        self.semobridge.converted_projected = converted_projected
        self.semobridge.converted_unprojected = converted_unprojected
        self.semobridge.converted_projected_mean = converted_projected.mean(dim=1)

    def train(self, vis=False):
        """Generic training loops."""
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            stop = self.after_epoch()
            if stop:
                break
        accuracy = self.after_train(vis=vis)
        return accuracy

    def before_train(self):
        self.set_model_mode("train", ["semobridge"])

        # Print model parameters
        print("Model parameters:")
        for name, param in self.semobridge.named_parameters():
            print(f"{name}: {param.shape} (requires_grad={param.requires_grad})")

        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        self.best_loss = torch.tensor(float("inf"))
        self.best_epoch = 0
        self.best_epoch_early_stopping = 0
        self.best_loss_early_stopping = torch.tensor(float("inf"))
        self.loss = torch.tensor(float("inf"))

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        with torch.no_grad():
            self.loss_fn = torch.nn.CrossEntropyLoss(
                label_smoothing=self.cfg.TRAIN.LABEL_SMOOTHING
            )
            self.loss_fn_no_reduction = torch.nn.CrossEntropyLoss(
                label_smoothing=self.cfg.TRAIN.LABEL_SMOOTHING, reduction="none"
            )

            self.targets_single = torch.arange(self.num_classes, dtype=torch.long).to(
                self.device
            )

            self.targets = self.targets_single.repeat(self.features.num_shots, 1)
            self.targets = self.targets.T
            self.targets = self.targets.flatten()
            self.targets = self.targets.to(self.device)

            self.val_params = {
                "smoothness": 0.01,
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": 1.0,
                "delta": 1.0,
                "lambda1": 0.5,
                "lambda2": 0.75,
                "lambda3": 0.1,
                "lambda4": 0.1,
            }
            # Use the init parameters from the cfg
            for param_name in self.val_params.keys():
                if hasattr(self.cfg.HP_SEARCH.PARAMS, param_name):
                    self.val_params[param_name] = self.cfg.HP_SEARCH.PARAMS[
                        param_name
                    ].INIT
                    print(f"Using {param_name} from cfg: {self.val_params[param_name]}")

            self.tfm = build_transform(self.cfg, is_train=True)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def bias_norm_loss(self, class_bias):
        # class_bias: [C, D]
        norms = class_bias.norm(dim=1)  # [C]
        mean_norm = norms.mean()
        loss = ((norms - mean_norm) ** 2).mean()  # FLOPS: C * C
        return loss

    def run_epoch(self):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = 1

        end = time.time()

        self.batch_idx = 0

        data_time.update(time.time() - end)

        # converted_prompts are per sample, not per class
        self.convert_final_prompts(self.features.few_shot_embeds)

        self.final_prompts_normed = F.normalize(
            self.semobridge.converted_projected_mean, dim=-1
        )

        converted_unprojected_mean = self.semobridge.converted_unprojected.mean(dim=1)
        converted_unprojected_flat = self.semobridge.converted_unprojected.flatten(
            start_dim=0, end_dim=1
        )

        converted_unprojected_flat_normed = F.normalize(
            converted_unprojected_flat, dim=-1
        )
        converted_unprojected_mean_normed = F.normalize(
            converted_unprojected_mean, dim=-1
        )

        converted_projected_flat = self.semobridge.converted_projected.flatten(
            start_dim=0, end_dim=1
        )
        converted_projected_flat_normed = F.normalize(converted_projected_flat, dim=-1)

        converted_projected_mean_normed = F.normalize(
            self.semobridge.converted_projected_mean, dim=-1
        )

        if "img_flat" in self.cfg.TRAIN.LOSSES:
            logits = (
                self.features.few_shot_embeds_flat_normed
                @ converted_projected_flat_normed.T
            ) * self.features.logit_scale
            img_flat = self.loss_fn(logits, self.targets_image)

        if "img_mean" in self.cfg.TRAIN.LOSSES:
            logits = (
                self.features.few_shot_embeds_flat_normed
                @ converted_projected_mean_normed.T
            ) * self.features.logit_scale
            img_mean = self.loss_fn(logits, self.targets)

        if "txte_mean" in self.cfg.TRAIN.LOSSES:
            logits = (
                converted_unprojected_mean_normed @ self.features.text_encoded_normed.T
            )
            txte_mean = self.loss_fn(logits, self.targets_single)

        if "txte_flat" in self.cfg.TRAIN.LOSSES:
            logits = (
                converted_unprojected_flat_normed @ self.features.text_encoded_normed.T
            )
            txte_flat = self.loss_fn(logits, self.targets)

        if "txtp_flat" in self.cfg.TRAIN.LOSSES:
            logits = (
                converted_projected_flat_normed @ self.features.text_projected_normed.T
            )
            txtp_flat = self.loss_fn(logits, self.targets)

        if "txtp_mean" in self.cfg.TRAIN.LOSSES:
            logits = self.final_prompts_normed @ self.features.text_projected_normed.T
            txtp_mean = self.loss_fn(logits, self.targets_single)

        if "consistency" in self.cfg.TRAIN.LOSSES:
            logits = converted_projected_flat_normed @ self.final_prompts_normed.T
            consistency = self.loss_fn(logits, self.targets)

        if "bias_norm" in self.cfg.TRAIN.LOSSES and self.cfg.TRAINER.SEMOBRIDGE.CSB:
            bias_norm = self.bias_norm_loss(self.semobridge.class_bias)

        loss_summary = {}

        # Calculate the total loss
        loss = 0

        image_loss_total = 0
        num_of_image_losses = 0
        if "img_flat" in self.cfg.TRAIN.LOSSES:
            image_loss_total += img_flat
            loss_summary["img_flat_loss"] = img_flat.item()
            num_of_image_losses += 1
        if "img_mean" in self.cfg.TRAIN.LOSSES:
            image_loss_total += img_mean
            loss_summary["img_mean_loss"] = img_mean.item()
            num_of_image_losses += 1

        text_loss_total = 0
        num_of_text_losses = 0
        if "txtp_flat" in self.cfg.TRAIN.LOSSES:
            text_loss_total += txtp_flat
            loss_summary["txtp_flat_loss"] = txtp_flat.item()
            num_of_text_losses += 1
        if "txtp_mean" in self.cfg.TRAIN.LOSSES:
            text_loss_total += txtp_mean
            loss_summary["txtp_mean_loss"] = txtp_mean.item()
            num_of_text_losses += 1
        if "txte_flat" in self.cfg.TRAIN.LOSSES:
            text_loss_total += txte_flat
            loss_summary["txte_flat_loss"] = txte_flat.item()
            num_of_text_losses += 1
        if "txte_mean" in self.cfg.TRAIN.LOSSES:
            text_loss_total += txte_mean
            loss_summary["txte_mean_loss"] = txte_mean.item()
            num_of_text_losses += 1
        loss_temp = (
            self.cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT[self.cfg.DATASET.NAME]
            * (image_loss_total / num_of_image_losses if num_of_image_losses > 0 else 1)
        ) + (
            (1 - self.cfg.TRAIN.IMAGE_TEXT_LOSS_WEIGHT[self.cfg.DATASET.NAME])
            * (text_loss_total / num_of_text_losses if num_of_text_losses > 0 else 1)
        )
        loss += loss_temp

        # loss += image_loss_total * self.cfg.TRAIN.IMAGE_LOSS_WEIGHT
        # loss += text_loss_total * self.cfg.TRAIN.TEXT_LOSS_WEIGHT

        if "consistency" in self.cfg.TRAIN.LOSSES:
            loss += consistency * self.cfg.TRAIN.CONS_LOSS_WEIGHT[self.cfg.DATASET.NAME]
            loss_summary["consistency_loss"] = consistency.item()
        if "bias_norm" in self.cfg.TRAIN.LOSSES and self.cfg.TRAINER.SEMOBRIDGE.CSB:
            loss += (
                bias_norm * self.cfg.TRAIN.BIAS_NORM_LOSS_WEIGHT[self.cfg.DATASET.NAME]
            )
            loss_summary["bias_norm_loss"] = bias_norm.item()

        loss_summary["total_loss"] = loss.item()

        self.last_loss = self.loss
        self.loss = loss
        self.model_backward_and_update(loss)

        # if (self.batch_idx + 1) == self.num_batches:
        self.update_lr()

        batch_time.update(time.time() - end)
        losses.update(loss_summary)

        # meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
        meet_freq = (self.epoch + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
        # only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
        # if meet_freq or only_few_batches:
        if meet_freq:
            nb_remain = 0
            nb_remain += self.num_batches - self.batch_idx - 1
            nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
            eta_seconds = batch_time.avg * nb_remain
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))

            info = []
            info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
            info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
            info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
            info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
            info += [f"{losses}"]
            info += [f"lr {self.get_current_lr():.4e}"]
            info += [f"eta {eta}"]
            print(" ".join(info))

        n_iter = self.epoch * self.num_batches + self.batch_idx
        for name, meter in losses.meters.items():
            self.write_scalar("train/" + name, meter.avg, n_iter)
        self.write_scalar("train/lr", self.get_current_lr(), n_iter)

        end = time.time()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0
            else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            self.convert_final_prompts(self.features.few_shot_embeds)
            final_prompts = self.semobridge.converted_projected_mean  # Shape [C, K, D]
            final_prompts_normed = F.normalize(final_prompts, dim=-1)

            val_embeds_converted_projected, val_embeds_converted_unprojected = (
                self.semobridge(self.features.val_embeds, use_class_bias=False)
            )
            val_embeds_converted_projected_normed = F.normalize(
                val_embeds_converted_projected, dim=-1
            )

            semobridge_logits = self.features.val_embeds_normed @ final_prompts_normed.T
            semobridge_conv_images_logits = (
                val_embeds_converted_projected_normed
                @ self.features.few_shot_embeds_mean_normed.T
            )
            semobridge_conv_prompt_logits = (
                val_embeds_converted_projected_normed @ self.final_prompts_normed.T
            )

            val_logits = self.blend_logits(
                self.features.few_shot_divergence,
                self.features.val_clip_logits,
                semobridge_logits,
                semobridge_conv_images_logits,
                semobridge_conv_prompt_logits,
                self.val_params,
            )

            pred = val_logits.argmax(dim=1)
            acc = (pred == self.features.val_labels).float().mean().item() * 100.0

            print(f"Epoch {self.epoch} validation accuracy: {acc:.2f}%")
            self.write_scalar("val/accuracy", acc, self.epoch)
            is_best = acc > self.best_result
            if is_best:
                self.best_result = acc
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=acc,
                    model_name="model-best.pth.tar",
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

        if self.loss < self.best_loss:
            self.best_loss = self.loss
            if self.cfg.TEST.FINAL_MODEL == "min_loss":
                self.best_epoch = self.epoch
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=None,
                    model_name="min_loss.pth.tar",
                )

        # Do early stopping if the loss does not improve over the threshold for the patience number of epochs
        if self.cfg.TRAIN.EARLY_STOPPING.ENABLED:
            if (
                self.loss + self.cfg.TRAIN.EARLY_STOPPING.THRESHOLD
                < self.best_loss_early_stopping
            ):
                self.best_loss_early_stopping = self.loss
                self.best_epoch_early_stopping = self.epoch

            # print(self.epoch - self.best_epoch_early_stopping)
            if (
                self.epoch - self.best_epoch_early_stopping
                >= self.cfg.TRAIN.EARLY_STOPPING.PATIENCE
            ):
                print(
                    f"Early stopping at epoch {self.epoch} with best loss {self.best_loss_early_stopping.item()} at epoch {self.best_epoch_early_stopping}"
                )
                return True

        return False  # No stopping

    def after_train(self, vis=False):
        print("Finish training")
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        print(
            "FINAL AVG_TEXT_EMBEDDING_LENGTH:",
            self.semobridge.avg_text_embedding_length.item(),
        )

        print("Calculating final prompts")

        with torch.no_grad():
            self.convert_final_prompts(self.features.few_shot_embeds)
            self.is_converted = True

        # if vis:
        #     self.generate_visualizations()

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir, model_file="model-best")
            elif self.cfg.TEST.FINAL_MODEL == "min_loss":
                print("Deploy the model with the minimum loss")
                self.load_model(self.output_dir, model_file="min_loss")
            else:
                print("Deploy the last-epoch model")
            accuracy = self.test()
        else:
            accuracy = None

        # Close writer
        self.close_writer()

        return accuracy

    def figure_intra_modal_vs_inter_modal_similarity(self):
        # classes_for_similarity_matrix = [""]

        if self.cfg.DATASET.NAME == "OxfordPets":
            classnames_to_show = [
                "abyssinian",
                "basset hound",
                "german shorthaired",
                "leonberger",
                "russian blue",
                "beagle",
                "bengal",
                "egyptian mau",
                "english setter",
                "english cocker spaniel",
            ]
        else:
            classnames_to_show = self.features.classnames[5:25]  # Show first 50 classes

        # We want to select some
        classes_to_show = torch.tensor(
            [
                self.features.classnames.index(cn)
                for cn in classnames_to_show
                if cn in self.features.classnames
            ],
            dtype=torch.long,
            device=self.device,
        )
        selected_classnames = [
            self.features.classnames[i] for i in classes_to_show.tolist()
        ]
        selected_num_classes = len(selected_classnames)

        # Select k test images per selected class
        k = 16
        selected_indices = []
        for class_id in classes_to_show.tolist():
            indices = torch.where(self.features.test_labels == class_id)[0]
            if len(indices) > k:
                indices = indices[:k]
            selected_indices.extend(indices.tolist())

        selected_test_labels = self.features.test_labels[selected_indices]
        selected_test_embeds_normed = self.features.test_embeds_normed[
            selected_indices
        ]  # [N', D]
        selected_text_projected_normed = self.features.text_projected_normed[
            classes_to_show
        ]  # [C, D]
        selected_few_shot_embeds_mean_normed = (
            self.features.few_shot_embeds_mean_normed[classes_to_show]
        )  # [C, D]

        # Convert the final prompts to the projected space
        converted_few_shot = self.semobridge(
            self.features.few_shot_embeds, use_class_bias=True, use_untrained=False
        )
        converted_few_shot_mean = converted_few_shot[0].mean(dim=1)  # [C, D]

        selected_converted_few_shot_mean = converted_few_shot_mean[
            classes_to_show
        ]  # [C, D]
        selected_converted_few_shot_mean_normed = F.normalize(
            selected_converted_few_shot_mean, dim=-1
        )  # [C, D]

        # Bridge test embeds with SeMoBridge
        bridged_test_embeds, _ = self.semobridge(
            self.features.test_embeds[selected_indices],
            use_class_bias=False,
            use_untrained=False,
        )
        # Normalize the bridged test embeds
        bridged_test_embeds_normed = F.normalize(bridged_test_embeds, dim=-1)

        # Gather image paths for the first selected image of each class
        class_images_x = []
        class_images_y = []
        for class_id in classes_to_show.tolist():
            for i in range(self.dm.train_loader_x.dataset.data_source.__len__()):
                if self.dm.train_loader_x.dataset.data_source[i].label == class_id:
                    idx = i
                    break

            path = self.dm.train_loader_x.dataset.data_source[idx].impath
            loaded_img = Image.open(path).convert("RGB")
            shortest_side = min(loaded_img.size)
            loaded_img = CenterCrop(shortest_side)(loaded_img)
            class_images_x.append(loaded_img)

            # Find the first occurrence of the class_id in test_labels
            idx = torch.where(self.features.test_labels == class_id)[0][0]
            path = self.dm.test_loader.dataset.data_source[idx].impath
            loaded_img = Image.open(path).convert("RGB")
            shortest_side = min(loaded_img.size)
            loaded_img = CenterCrop(shortest_side)(loaded_img)
            class_images_y.append(loaded_img)

        # Heatmap of cosine similarity between final prompts and text embeddings
        clip_logits = (
            selected_test_embeds_normed @ selected_text_projected_normed.T
        )  # [N', C]
        intra_modal_image_logits = (
            selected_test_embeds_normed @ selected_few_shot_embeds_mean_normed.T
        )
        fFhat_logits = (
            selected_test_embeds_normed @ selected_converted_few_shot_mean_normed.T
        )
        fhatF_logits = (
            bridged_test_embeds_normed @ selected_few_shot_embeds_mean_normed.T
        )

        # Make a CxC matrix that shows average cosine similarity between true and predicted classes (values between 0 and 1)
        clip_predictions = torch.zeros(
            selected_num_classes,
            selected_num_classes,
            device=self.device,
            dtype=torch.float32,
        )
        intra_modal_predictions = torch.zeros(
            selected_num_classes,
            selected_num_classes,
            device=self.device,
            dtype=torch.float32,
        )
        fFhat_predictions = torch.zeros(
            selected_num_classes,
            selected_num_classes,
            device=self.device,
            dtype=torch.float32,
        )
        fhatF_predictions = torch.zeros(
            selected_num_classes,
            selected_num_classes,
            device=self.device,
            dtype=torch.float32,
        )

        for i in range(selected_test_embeds_normed.shape[0]):
            true_label_class = selected_test_labels[i]
            true_label = classes_to_show.tolist().index(true_label_class)

            clip_logits_i = clip_logits[i]
            intra_modal_logits = intra_modal_image_logits[i]
            fFhat_logits_i = fFhat_logits[i]
            fhatF_logits_i = fhatF_logits[i]

            # Get the cosine similarity
            clip_predictions[true_label] += clip_logits_i
            intra_modal_predictions[true_label] += intra_modal_logits
            fFhat_predictions[true_label] += fFhat_logits_i
            fhatF_predictions[true_label] += fhatF_logits_i

        # Normalize the predictions by the number of samples in each class
        clip_predictions /= selected_test_embeds_normed.shape[0]
        intra_modal_predictions /= selected_test_embeds_normed.shape[0]
        fFhat_predictions /= selected_test_embeds_normed.shape[0]
        fhatF_predictions /= selected_test_embeds_normed.shape[0]

        # Normalize the logits to be min 0 and max 1
        clip_predictions = (clip_predictions - clip_predictions.min()) / (
            clip_predictions.max() - clip_predictions.min()
        )
        intra_modal_predictions = (
            intra_modal_predictions - intra_modal_predictions.min()
        ) / (intra_modal_predictions.max() - intra_modal_predictions.min())
        fFhat_predictions = (fFhat_predictions - fFhat_predictions.min()) / (
            fFhat_predictions.max() - fFhat_predictions.min()
        )
        fhatF_predictions = (fhatF_predictions - fhatF_predictions.min()) / (
            fhatF_predictions.max() - fhatF_predictions.min()
        )

        # Create a subplot with two heatmaps: intra-modal (left) and semobridge predictions (right)
        fig = make_subplots(
            rows=1,
            cols=3,
            # subplot_titles=(
            #     # r"$\Large\text{CLIP}~~\mathbf f^\mathrm{proj}\mathbf T^\mathrm{proj}$",
            #     r"$\Large\text{Intra-modal}~~\mathbf f^\mathrm{proj}\mathbf F^\mathrm{proj}$",
            #     r"$\Large\text{SeMoBridge}~~\hat{\mathbf f}^\mathrm{proj}\mathbf F^\mathrm{proj}$",
            #     r"$\Large\text{SeMoBridge}~~\mathbf f^\mathrm{proj}\hat{\mathbf F}^\mathrm{proj}$",
            # ),
            subplot_titles=(
                # r"$\Large\text{CLIP}~~\mathbf f^\mathrm{proj}\mathbf T^\mathrm{proj}$",
                r"$\Huge\text{Intra-modal}~~\mathbf f^\mathrm{proj}\mathbf F^{\mathrm{proj}\top}$",
                r"$\Huge\text{Ours}~~{\mathbf f}^\mathrm{proj}\mathbf F^{\mathrm{proj}\top}$",
                r"$\Huge\text{Ours}~~\mathbf f^\mathrm{proj}{\mathbf F}^{\mathrm{proj}\top}$",
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.06,
        )

        # Numbers from 0 to selected_num_classes - 1
        axis_labels = [str(i) for i in range(selected_num_classes)]

        # # CLIP similarity heatmap (left)
        # fig.add_trace(
        #     go.Heatmap(
        #         z=clip_predictions.cpu(),
        #         x=axis_labels,
        #         y=axis_labels,
        #         zmin=0,
        #         # zmax=100,
        #         colorscale="Purples",
        #         hoverongaps=False,
        #         showscale=False,
        #     ),
        #     row=1,
        #     col=1,
        # )

        # Intra-modal image similarity heatmap
        fig.add_trace(
            go.Heatmap(
                z=intra_modal_predictions.cpu(),
                x=axis_labels,
                y=axis_labels,
                zmin=0,
                # zmax=100,
                colorscale="Purples",
                hoverongaps=False,
                showscale=False,
            ),
            row=1,
            col=1,
        )

        # Semobridge similarity heatmap (left)
        fig.add_trace(
            go.Heatmap(
                z=fhatF_predictions.cpu(),
                x=axis_labels,
                y=axis_labels,
                zmin=0,
                # zmax=100,
                colorscale="Purples",
                hoverongaps=False,
                yaxis="y2",
                showscale=False,
            ),
            row=1,
            col=2,
        )

        # Semobridge similarity heatmap (right)
        fig.add_trace(
            go.Heatmap(
                z=fFhat_predictions.cpu(),
                x=axis_labels,
                y=axis_labels,
                zmin=0,
                # zmax=100,
                colorscale="Purples",
                hoverongaps=False,
                yaxis="y2",
                # Color scale for the right heatmap with ticks at 0, 10, ..., 100
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="",
                        font=dict(size=24, family="Times New Roman"),
                    ),
                    # tickvals=list(range(0, 110, 10)),
                    # ticktext=[str(i) for i in range(0, 110, 10)],
                ),
            ),
            row=1,
            col=3,
        )

        # Show axes with ticks
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(
                    showticklabels=True,
                    tickangle=0,
                    row=row,
                    col=col,
                )
                fig.update_yaxes(
                    showticklabels=True,
                    tickangle=0,
                    row=row,
                    col=col,
                )

        # # Hide x axes classnames for the left heatmap
        # fig.update_xaxes(
        #     showticklabels=False,
        #     row=1,
        #     col=1,
        #     layer="below traces",
        #     range=[-1.7, selected_num_classes],
        # )
        # fig.update_yaxes(
        #     showticklabels=False,
        #     row=1,
        #     col=1,
        #     layer="below traces",
        #     range=[-1.6, selected_num_classes],
        # )

        # # Hide y axis classnames for the right heatmap
        # fig.update_xaxes(
        #     showticklabels=False,
        #     row=1,
        #     col=2,
        #     layer="below traces",
        #     range=[-1.0, selected_num_classes + 1.5],
        # )
        # fig.update_yaxes(
        #     showticklabels=False,
        #     row=1,
        #     col=2,
        #     layer="below traces",
        #     range=[-1.6, selected_num_classes],
        # )

        # # Add border to the heatmaps
        # fig.update_xaxes(
        #     showline=False,
        #     layer="below traces",
        # )
        # fig.update_yaxes(
        #     showline=False,
        #     layer="below traces",
        # )

        fig.update_layout(
            # title=f"{self.cfg.DATASET.NAME}",
            xaxis_title="",
            yaxis_title="Query Image Class",
            font=dict(size=44, family="Times New Roman", color="black", weight=500),
            # Remove margin and padding
            margin=dict(l=5, r=2, t=80, b=150),
        )

        # Increase space between plot and axis title
        for col in range(1, 4):
            fig.update_xaxes(title_standoff=20, row=1, col=col)
            fig.update_yaxes(title_standoff=20, row=1, col=col)

        # Increase space between axis numbers and heatmaps
        for col in range(1, 4):
            fig.update_xaxes(ticklabelposition="outside", row=1, col=col)
            fig.update_yaxes(ticklabelposition="outside", row=1, col=col)

        # Add borders to the heatmaps
        for col in range(1, 4):
            fig.update_xaxes(
                showline=True,
                linecolor="black",
                linewidth=2,
                layer="below traces",
                row=1,
                col=col,
                mirror=True,  # Mirror the x-axis line
            )
            fig.update_yaxes(
                showline=True,
                linecolor="black",
                linewidth=2,
                layer="below traces",
                row=1,
                col=col,
                mirror=True,  # Mirror the y-axis line
            )

        # Increase space between subplot titles and heatmaps
        fig.layout.annotations[0].update(yshift=13)
        fig.layout.annotations[1].update(yshift=13)
        fig.layout.annotations[2].update(yshift=13)

        # Add text annotation below the two heatmaps as X-axis title
        fig.add_annotation(
            text="Few-shot Image Class",
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            x=0.5,
            y=-0.31,
            showarrow=False,
            font=dict(size=48, family="Times New Roman", color="black", weight=500),
        )

        # Add text annotation to below the colorbar
        fig.add_annotation(
            text="Sim.",
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="bottom",
            x=1.075,
            y=-0.2,
            showarrow=False,
            font=dict(size=34, family="Times New Roman", color="black", weight=500),
        )

        # Remove plot background color
        fig.update_layout(plot_bgcolor="white")

        # fig.layout.annotations[0].update(
        #     font=dict(family="Times New Roman", size=64, weight="normal")
        # )
        # fig.layout.annotations[1].update(
        #     font=dict(family="Times New Roman", size=64, weight="normal")
        # )

        # fig.update_xaxes(tickangle=45)
        fig_path = os.path.join(
            self.output_dir, "intra-modal-vs-inter-modal-similarity.pdf"
        )
        # fig.write_html(fig_path)
        # fig.write_image("trash.pdf")  # Because of plotly bug
        # time.sleep(1)
        fig.write_image(fig_path, width=1700, height=650)
        print(f"Saved intra-modal vs inter-modal similarity heatmap to {fig_path}")

    def figure_cosine_similarity_histogram(self):
        save_path = os.path.join(
            "preprocessed",
            f"{self.cfg.MODEL.BACKBONE.NAME.replace("/", "")}_{self.cfg.TRAINER.SEMOBRIDGE.PREC}",
            self.cfg.DATASET.DIRECTORY,
            f"aphotoofa.pt",
        )
        if not os.path.exists(save_path):
            print(f"File {save_path} does not exist, embedding...")
            # We need to load CLIP for this
            self.features.load_clip_if_not_loaded()
            texts = self.features.load_class_text_prompts("aphotoofa")
            text_projected, text_projected_normed, text_encoded, text_encoded_normed = (
                self.features.embed_class_text_prompts(texts)
            )
            torch.save(
                [
                    text_projected,
                    text_projected_normed,
                    text_encoded,
                    text_encoded_normed,
                ],
                save_path,
            )
        else:
            print(f"Loading {save_path}")
            text_projected, text_projected_normed, text_encoded, text_encoded_normed = (
                torch.load(save_path)
            )
        text_projected_clip_normed = F.normalize(text_projected, dim=-1)

        # Pick a subset of test embeds and labels
        test_subset_indices = []
        # k_per_class = self.features.num_shots
        # k_per_class = 1

        # for class_id in self.features.test_labels.unique():
        #     class_indices = torch.where(self.features.test_labels == class_id)[0]
        #     if class_indices.numel() > k_per_class:
        #         class_indices = class_indices[:k_per_class]
        #     test_subset_indices.append(class_indices)
        # test_subset_indices = torch.cat(test_subset_indices)

        # ALL
        test_subset_indices = torch.arange(
            self.features.test_embeds.shape[0], device=self.device
        )

        test_embeds_subset = self.features.test_embeds[test_subset_indices]
        test_embeds_normed_subset = self.features.test_embeds_normed[
            test_subset_indices
        ]
        test_labels_subset = self.features.test_labels[test_subset_indices]

        image_image_sims = (
            test_embeds_normed_subset @ self.features.few_shot_embeds_mean_normed.T
        )

        # text_only = self.text_sims.flatten().cpu().numpy()
        # image_only = self.image_sims.flatten().cpu().numpy()

        image_text_sims = test_embeds_normed_subset @ text_projected_clip_normed.T

        unpaired_image_only = image_image_sims[
            test_labels_subset.unsqueeze(1) != self.targets_single.unsqueeze(0)
        ]  # [N', K * (C - 1)]
        unpaired_image_only = unpaired_image_only.flatten()
        paired_image_only = image_image_sims[
            test_labels_subset.unsqueeze(1) == self.targets_single.unsqueeze(0)
        ]  # [N', 1]
        paired_image_only = paired_image_only.flatten()

        unpaired_image_text = image_text_sims[
            test_labels_subset.unsqueeze(1) != self.targets_single.unsqueeze(0)
        ]  # [N', K * (C - 1)]
        unpaired_image_text = unpaired_image_text.flatten()
        paired_image_text = image_text_sims[
            test_labels_subset.unsqueeze(1) == self.targets_single.unsqueeze(0)
        ]  # [N', 1]
        paired_image_text = paired_image_text.flatten()

        # Create untrained bridged few-shot embeds
        bridged_few_shot_embeds_untrained = self.semobridge(
            self.features.few_shot_embeds, use_class_bias=True, use_untrained=True
        )[0]
        bridged_few_shot_embeds_untrained = bridged_few_shot_embeds_untrained.mean(
            dim=1
        )  # [C, D]
        bridged_few_shot_embeds_untrained_normed = F.normalize(
            bridged_few_shot_embeds_untrained, dim=-1
        )

        bridged_test_untrained_sims = test_embeds_normed_subset @ (
            bridged_few_shot_embeds_untrained_normed.T
        )
        unpaired_untrained_bridged_test = bridged_test_untrained_sims[
            test_labels_subset.unsqueeze(1) != self.targets_single.unsqueeze(0)
        ]
        unpaired_untrained_bridged_test = unpaired_untrained_bridged_test.flatten()
        paired_untrained_bridged_test = bridged_test_untrained_sims[
            test_labels_subset.unsqueeze(1) == self.targets_single.unsqueeze(0)
        ]
        paired_untrained_bridged_test = paired_untrained_bridged_test.flatten()

        # Create trained bridged test embeds
        bridged_few_shot_embeds = self.semobridge(
            self.features.few_shot_embeds, use_class_bias=True, use_untrained=False
        )[0]
        bridged_few_shot_embeds = bridged_few_shot_embeds.mean(dim=1)  # [C, D]
        bridged_few_shot_embeds_normed = F.normalize(bridged_few_shot_embeds, dim=-1)
        bridged_test_sims = test_embeds_normed_subset @ (
            bridged_few_shot_embeds_normed.T
        )

        unpaired_bridged_test = bridged_test_sims[
            test_labels_subset.unsqueeze(1) != self.targets_single.unsqueeze(0)
        ]  # [N', K * (C - 1)]
        unpaired_bridged_test = unpaired_bridged_test.flatten()
        paired_bridged_test = bridged_test_sims[
            test_labels_subset.unsqueeze(1) == self.targets_single.unsqueeze(0)
        ]  # [N', 1]
        paired_bridged_test = paired_bridged_test.flatten()

        # Reduce the cosine similarity distribution resolution for smaller PDF size
        # We want to sample the distribution with 100 points
        min_sim = -0.1
        max_sim = 1.11

        # Calculate num_bins so that we have an integer number of bins
        num_bins = 250

        # unpaired_image_only = (
        #     torch.histc(unpaired_image_only, bins=num_bins, min=min_sim, max=max_sim)
        #     .cpu()
        #     .numpy()
        # )
        # paired_image_only = (
        #     torch.histc(paired_image_only, bins=num_bins, min=min_sim, max=max_sim)
        #     .cpu()
        #     .numpy()
        # )
        # unpaired_image_text = (
        #     torch.histc(unpaired_image_text, bins=num_bins, min=min_sim, max=max_sim)
        #     .cpu()
        #     .numpy()
        # )
        # paired_image_text = (
        #     torch.histc(paired_image_text, bins=num_bins, min=min_sim, max=max_sim)
        #     .cpu()
        #     .numpy()
        # )
        # unpaired_image_converted_untrained = (
        #     torch.histc(
        #         unpaired_image_converted_untrained,
        #         bins=num_bins,
        #         min=min_sim,
        #         max=max_sim,
        #     )
        #     .cpu()
        #     .numpy()
        # )
        # paired_image_converted_untrained = (
        #     torch.histc(
        #         paired_image_converted_untrained,
        #         bins=num_bins,
        #         min=min_sim,
        #         max=max_sim,
        #     )
        #     .cpu()
        #     .numpy()
        # )
        # unpaired_image_converted = (
        #     torch.histc(
        #         unpaired_image_converted, bins=num_bins, min=min_sim, max=max_sim
        #     )
        #     .cpu()
        #     .numpy()
        # )
        # paired_image_converted = (
        #     torch.histc(paired_image_converted, bins=num_bins, min=min_sim, max=max_sim)
        #     .cpu()
        #     .numpy()
        # )

        # Plot with Plotly
        fig = go.Figure()

        # marker.pattern.shape = ['', '/', '\\', 'x', '-', '|', '+', '.']
        # fig.add_trace(
        #     go.Histogram(
        #         x=text_only,
        #         name="Text-Text",
        #         histnorm="probability density",
        #         marker=dict(color="blue", pattern=dict(shape="/")),
        #         opacity=0.6,
        #     )
        # )

        min_sim = min_sim

        unpaired_image_only = unpaired_image_only.cpu()
        paired_image_only = paired_image_only.cpu()
        unpaired_image_text = unpaired_image_text.cpu()
        paired_image_text = paired_image_text.cpu()
        unpaired_untrained_bridged_test = unpaired_untrained_bridged_test.cpu()
        paired_untrained_bridged_test = paired_untrained_bridged_test.cpu()
        unpaired_bridged_test = unpaired_bridged_test.cpu().numpy()
        paired_bridged_test = paired_bridged_test.cpu().numpy()

        xbins = dict(
            start=min_sim,
            end=max_sim,
            size=(max_sim - min_sim) / num_bins,
        )

        print("Plotting cosine similarity distributions...")

        fig.add_trace(
            go.Histogram(
                x=unpaired_image_only,
                name=r"$\text{Intra-modal}~~ \mathbf F\mathbf F~~\text{Unpaired}$",
                histnorm="probability density",
                marker=dict(color="orange"),
                # marker=dict(color="orange", pattern=dict(shape="\\")),
                opacity=0.6,
                xbins=xbins,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=paired_image_only,
                name=r"$\text{Intra-modal}~~ \mathbf F\mathbf F~~\text{Paired}$",
                histnorm="probability density",
                marker=dict(color="yellow"),
                # marker=dict(color="yellow", pattern=dict(shape="\\")),
                opacity=0.6,
                xbins=xbins,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=unpaired_image_text,
                name=r"$\text{CLIP}~~ \mathbf F\mathbf T_\mathrm{proj}~~\text{Unpaired}$",
                histnorm="probability density",
                marker=dict(color="green"),
                # marker=dict(color="green", pattern=dict(shape="-")),
                opacity=0.6,
                xbins=xbins,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=paired_image_text,
                name=r"$\text{CLIP}~~ \mathbf F\mathbf T_\mathrm{proj}~~\text{Paired}$",
                histnorm="probability density",
                marker=dict(color="olive"),
                # marker=dict(color="olive", pattern=dict(shape="-")),
                opacity=0.6,
                xbins=xbins,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=unpaired_untrained_bridged_test,
                name=r"$\text{SeMoBridge}~~ \mathbf F\hat{\mathbf F}_\mathrm{proj}~~\text{Unpaired}$",
                histnorm="probability density",
                marker=dict(color="purple"),
                # marker=dict(color="purple", pattern=dict(shape=".")),
                opacity=0.6,
                xbins=xbins,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=paired_untrained_bridged_test,
                name=r"$\text{SeMoBridge}~~ \mathbf F\hat{\mathbf F}_\mathrm{proj}~~\text{Paired}$",
                histnorm="probability density",
                marker=dict(color="red"),
                # marker=dict(color="red", pattern=dict(shape=".")),
                opacity=0.6,
                xbins=xbins,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=unpaired_bridged_test,
                name=r"$\text{SeMoBridge-T}~~ \mathbf F\hat{\mathbf F}_\mathrm{proj}~~\text{Unpaired}$",
                histnorm="probability density",
                marker=dict(color="DodgerBlue"),
                # marker=dict(color="lightblue", pattern=dict(shape=".")),
                opacity=0.6,
                xbins=xbins,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=paired_bridged_test,
                name=r"$\text{SeMoBridge-T}~~ \mathbf F\hat{\mathbf F}_\mathrm{proj}~~\text{Paired}$",
                histnorm="probability density",
                marker=dict(color="blue"),
                # marker=dict(color="blue", pattern=dict(shape=".")),
                opacity=0.6,
                xbins=xbins,
            )
        )

        fig.update_layout(
            barmode="overlay",
            # title=f"Cosine Similarity Distributions by Modalities, {self.cfg.DATASET.NAME} {self.cfg.DATASET.NUM_SHOTS} shots, {self.cfg.MODEL.BACKBONE.NAME}",
            xaxis_title="Cosine Similarity",
            yaxis_title="Sample Density",
            bargap=0.0,
            margin=dict(l=0, r=0, t=0, b=100),  # Remove margin
            font=dict(
                size=28,
                family="Times New Roman",
                color="black",
                weight=500,
            ),
            # Make background white
            plot_bgcolor="white",
            paper_bgcolor="white",
            # Add outline to the plot
            xaxis=dict(
                showline=True,
                linecolor="black",
                linewidth=2,
                mirror=True,  # Mirror the outline on both sides
            ),
            yaxis=dict(
                showline=True,
                linecolor="black",
                linewidth=2,
                mirror=True,  # Mirror the outline on both sides
            ),
        )

        # Add grey line to x=0
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=1,
            line=dict(color="grey", width=2, dash="dash"),
            xref="x",
            yref="paper",
        )

        # Move legend to top right INSIDE the plot, vertically
        fig.update_layout(
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,1.0)",
            ),
        )

        print("Saving cosine similarity distributions...")
        fig_path = os.path.join(self.output_dir, "cosine_similarity_distributions.pdf")
        fig.write_image(fig_path, width=800, height=500)
        print(f"Saved cosine similarity distributions to {fig_path}")
        fig_path = fig_path.replace(".pdf", ".png")
        fig.write_image(fig_path, width=800, height=500)
        print(f"Saved cosine similarity distributions to {fig_path}")

    def check_bridged_few_shot_token_words(self):
        # Bridge the few-shots
        few_shot_embeds, few_shot_bridged_eos = self.semobridge(
            self.features.few_shot_embeds, use_class_bias=True, use_untrained=True
        )

        few_shot_bridged_eos_mean = few_shot_bridged_eos.mean(dim=1)  # [C, D]

        tokenizer = _Tokenizer()

        topk = 10

        # Get all CLIP tokenizer words
        self.features.load_clip_if_not_loaded()
        token_embedding = self.features.model.clip_model.token_embedding.weight
        print(f"Size of token embedding: {token_embedding.shape}")

        distance = torch.cdist(few_shot_bridged_eos_mean, token_embedding)
        print(f"Size of distance matrix: {distance.shape}")
        sorted_idxs = torch.argsort(distance, dim=1)
        sorted_idxs = sorted_idxs[:, :topk]

        for m, idxs in enumerate(sorted_idxs):
            words = [tokenizer.decoder[idx.item()] for idx in idxs]
            dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
            print(f"{m+1}: {words} {dist}")

    def generate_visualizations(self):
        # Generate visualizations

        #self.check_bridged_few_shot_token_words()
        self.figure_intra_modal_vs_inter_modal_similarity()
        self.figure_cosine_similarity_histogram()

        # fig_path = os.path.join(self.output_dir, "cosine_similarity_distributions.png")
        # fig.write_image(fig_path)
        # print(f"Saved cosine similarity distributions to {fig_path}")

        # ########################### UMAP ####################################
        # # UMAP visualization of the image embeddings, text embeddings, and converted prompts
        # from umap import UMAP
        # import numpy as np
        # import random

        # # Define colors for each class
        # class_colors = [
        #     f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(self.num_classes)
        # ]

        # features = []
        # texts = []
        # colors = []
        # symbols = []

        # # Image embeddings
        # features.append(self.image_embeds_mean.cpu().numpy())
        # texts += [
        #     f"{c} (Image)"
        #     for c in self.semobridge.classnames
        #     # for _ in range(self.features.num_shots)
        # ]
        # colors += [
        #     class_colors[c]
        #     for c in range(self.num_classes)
        #     # for _ in range(self.features.num_shots)
        # ]
        # symbols += [f"square" for _ in range(self.num_classes)]
        # # symbols += [f"square" for _ in range(self.num_classes * self.features.num_shots)]

        # # Text embeddings
        # features.append(self.semobridge.text_projected.cpu().numpy())
        # texts += [f"{c} (Text)" for c in self.semobridge.classnames]
        # colors += [class_colors[c] for c in range(self.num_classes)]
        # symbols += [f"circle" for _ in range(self.num_classes)]

        # # Converted prompts
        # features.append(
        #     # self.semobridge.converted_projected.flatten(start_dim=0, end_dim=1)
        #     converted_projected_untrained_mean.cpu().numpy()
        # )
        # texts += [
        #     f"{c} (Converted Untrained)"
        #     for c in self.semobridge.classnames
        #     # for _ in range(self.features.num_shots)
        # ]
        # colors += [
        #     class_colors[c]
        #     for c in range(self.num_classes)
        #     # for _ in range(self.features.num_shots)
        # ]
        # symbols += [
        #     f"diamond"
        #     for _ in range(self.num_classes)
        #     # for _ in range(self.features.num_shots)
        # ]

        # # Concatenate all features, texts, and colors
        # features = np.concatenate(features, axis=0)
        # texts = np.array(texts)
        # colors = np.array(colors)
        # symbols = np.array(symbols)

        # # # Generate colors from 3D PCA
        # # from sklearn.decomposition import PCA

        # # pca = PCA(n_components=3)
        # # pca_features = pca.fit_transform(features)
        # # # Normalize PCA features to [0, 1] range for color mapping
        # # pca_features = (pca_features - pca_features.min(axis=0)) / (
        # #     pca_features.max(axis=0) - pca_features.min(axis=0)
        # # )
        # # # Convert PCA features to RGB colors
        # # colors = [
        # #     f"rgb({int(255 * r)}, {int(255 * g)}, {int(255 * b)})"
        # #     for r, g, b in pca_features
        # # ]

        # # Apply UMAP
        # umap_model = UMAP(
        #     n_neighbors=5,
        #     # n_neighbors=self.num_classes,
        #     min_dist=0.5,
        #     metric="cosine",
        #     random_state=42,
        #     verbose=True,
        # )
        # umap_embeddings = umap_model.fit_transform(features)
        # print(f"UMAP embeddings shape: {umap_embeddings.shape}")

        # # Create a DataFrame for Plotly
        # import pandas as pd

        # df = pd.DataFrame(
        #     {
        #         "x": umap_embeddings[:, 0],
        #         "y": umap_embeddings[:, 1],
        #         "text": texts,
        #         "color": colors,
        #         "shape": symbols,
        #     }
        # )
        # # Create a scatter plot with Plotly
        # fig = go.Figure(
        #     data=go.Scatter(
        #         x=df["x"],
        #         y=df["y"],
        #         mode="markers",
        #         opacity=0.75,
        #         marker=dict(
        #             color=df["color"],
        #             size=5,
        #             line=dict(width=0.5, color="DarkSlateGrey"),
        #             symbol=df["shape"],
        #         ),
        #         text=df["text"],
        #         hoverinfo="text",
        #     )
        # )
        # fig.update_layout(
        #     title=f"UMAP Visualization of Image Embeddings, Text Embeddings, and Converted Prompts (Untrained), {self.cfg.DATASET.NAME} {self.cfg.DATASET.NUM_SHOTS} shots, {self.cfg.MODEL.BACKBONE.NAME}",
        #     xaxis_title="UMAP Dimension 1",
        #     yaxis_title="UMAP Dimension 2",
        #     width=1600,
        #     height=1600,
        #     font=dict(
        #         size=14,
        #     ),
        # )
        # fig_path = os.path.join(self.output_dir, "umap_visualization.html")
        # fig.write_html(fig_path)
        # print(f"Saved UMAP visualization to {fig_path}")

    @torch.no_grad()
    def test(self, split=None, vis=False, eval_only=False):
        """A generic testing pipeline."""
        self.set_model_mode("eval")

        self.evaluator.reset()

        if self.num_classes != len(self.dm.dataset.classnames):
            self.convert_to_ood()

        if eval_only:
            self.before_train()
            self.convert_final_prompts(self.features.few_shot_embeds)

        if vis:
            self.generate_visualizations()
            return

        # Normalize prompts
        final_prompts = self.semobridge.converted_projected_mean  # Shape [C, K, D]
        # converted_projected_flat = final_prompts.flatten(start_dim=0, end_dim=1)  # Shape [C*K, D]
        final_prompts_normed = F.normalize(final_prompts, dim=-1)

        # Print CLIP's zero-shot accuracy
        clip_pred = self.features.test_clip_logits.argmax(dim=1)
        clip_zs_accuracy = (
            clip_pred == self.features.test_labels
        ).float().mean().item() * 100.0
        print(f"CLIP's zero-shot accuracy: {clip_zs_accuracy:.2f}%")

        semobridge_logits = self.features.val_embeds_normed @ final_prompts_normed.T

        val_embeds_converted_projected, val_embeds_converted_unprojected = (
            self.semobridge(self.features.val_embeds, use_class_bias=False)
        )
        val_embeds_converted_projected_normed = F.normalize(
            val_embeds_converted_projected, dim=-1
        )

        semobridge_conv_images_logits = (
            val_embeds_converted_projected_normed
            @ self.features.few_shot_embeds_mean_normed.T
        )

        semobridge_conv_prompt_logits = (
            val_embeds_converted_projected_normed @ final_prompts_normed.T
        )

        direct_intra_modal_logits = (
            self.features.val_embeds_normed
            @ self.features.few_shot_embeds_mean_normed.T
        )

        # Calculate results for all logits
        acc = (
            direct_intra_modal_logits.argmax(dim=1) == self.features.val_labels
        ).float().mean().item() * 100.0
        print(f"f^proj <> F^proj: {acc:.2f}%")
        acc = (
            semobridge_logits.argmax(dim=1) == self.features.val_labels
        ).float().mean().item() * 100.0
        print(f"f^proj <> Fhat^proj: {acc:.2f}%")

        acc = (
            semobridge_conv_images_logits.argmax(dim=1) == self.features.val_labels
        ).float().mean().item() * 100.0
        print(f"fhat^proj <> F^proj: {acc:.2f}%")
        acc = (
            semobridge_conv_prompt_logits.argmax(dim=1) == self.features.val_labels
        ).float().mean().item() * 100.0
        print(f"fhat^proj <> Fhat^proj: {acc:.2f}%")

        results = None

        # del self.semobridge
        del val_embeds_converted_projected
        del val_embeds_converted_projected_normed
        del val_embeds_converted_unprojected
        del final_prompts
        # del final_prompts_normed

        # Clear cache to free memory
        self.features.clear_memory()

        global best_logits
        global best_acc
        global best_params

        best_logits = None
        best_acc = 0.0

        param_names = []

        # Check if any of z2, z3, z4 are in self.cfg.TEST.LOGITS
        if (
            "z2" in self.cfg.TEST.LOGITS
            or "z3" in self.cfg.TEST.LOGITS
            or "z4" in self.cfg.TEST.LOGITS
        ):
            param_names.append("smoothness")

        if "z1" in self.cfg.TEST.LOGITS:
            param_names.append("alpha")
        if "z2" in self.cfg.TEST.LOGITS:
            param_names.append("beta")
        if "z3" in self.cfg.TEST.LOGITS:
            param_names.append("gamma")
        if "z4" in self.cfg.TEST.LOGITS:
            param_names.append("delta")

        if "z1" in self.cfg.TEST.LOGITS:
            param_names.append("lambda1")
        if "z2" in self.cfg.TEST.LOGITS:
            param_names.append("lambda2")
        if "z3" in self.cfg.TEST.LOGITS:
            param_names.append("lambda3")
        if "z4" in self.cfg.TEST.LOGITS:
            param_names.append("lambda4")

        # param_names = [
        #     "smoothness",
        #     "alpha",
        #     "beta",
        #     "gamma",
        #     "delta",
        #     "lambda1",
        #     "lambda2",
        #     "lambda3",
        #     "lambda4",
        # ]

        for param_name in param_names:
            if param_name not in self.cfg.HP_SEARCH.PARAMS:
                raise ValueError(f"Parameter {param_name} not found in config.")

            print(
                f"{param_name} {self.cfg.HP_SEARCH.PARAMS[param_name].START} to {self.cfg.HP_SEARCH.PARAMS[param_name].MAX}"
            )

        def objective(trial):
            global best_acc
            global best_params

            params = {}
            for param_name in param_names:
                if param_name not in self.cfg.HP_SEARCH.PARAMS:
                    raise ValueError(f"Parameter {param_name} not found in config.")

                params[param_name] = trial.suggest_float(
                    param_name,
                    self.cfg.HP_SEARCH.PARAMS[param_name].START,
                    self.cfg.HP_SEARCH.PARAMS[param_name].MAX,
                    # step=self.cfg.HP_SEARCH.PARAMS[param_name].STEP_SIZE,
                )
            with torch.no_grad():
                blended_logits = self.blend_logits(
                    self.features.few_shot_divergence,
                    self.features.val_clip_logits,
                    semobridge_logits,
                    semobridge_conv_images_logits,
                    semobridge_conv_prompt_logits,
                    params,
                )
                pred = blended_logits.argmax(dim=1)
                acc = (pred == self.features.val_labels).float().mean().item() * 100.0

                if acc > best_acc:
                    best_acc = acc
                    best_params = params

                    params_string = ", ".join([f"{p:.4f}" for p in params.values()])
                    print(
                        f"New best setting at {trial.number}, accuracy: {best_acc:.2f}% with {params_string} ({params.keys()})"
                    )

            return acc

        set_random_seed(self.cfg.SEED)

        sampler = TPESampler(
            seed=self.cfg.SEED
        )  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.cfg.HP_SEARCH.N_TRIALS,
            # n_trials=50,
            n_jobs=1,  # To be deterministic, use 1 job only
            show_progress_bar=True,
        )

        # Process blended output with confidence
        print("Final test on test set with best settings:")
        # Convert the test embeddings to text embeddings
        test_embeds_converted_projected, test_embeds_converted_unprojected = (
            self.semobridge(self.features.test_embeds, use_class_bias=False)
        )

        # Normalize embeds
        test_embeds_converted_projected_normed = F.normalize(
            test_embeds_converted_projected, dim=-1
        )

        # Compare the test embeddings with text embeddings (vanilla CLIP)
        semobridge_logits = self.features.test_embeds_normed @ final_prompts_normed.T
        semobridge_conv_images_logits = (
            test_embeds_converted_projected_normed
            @ self.features.few_shot_embeds_mean_normed.T
        )
        semobridge_conv_prompt_logits = (
            test_embeds_converted_projected_normed @ final_prompts_normed.T
        )

        final_logits = self.blend_logits(
            self.features.few_shot_divergence,
            self.features.test_clip_logits,
            semobridge_logits,
            semobridge_conv_images_logits,
            semobridge_conv_prompt_logits,
            best_params,
        )

        self.evaluator.reset()
        self.evaluator.process(final_logits.to(self.device), self.features.test_labels)
        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def blend_logits(
        self,
        few_shot_divergence,
        clip_logits,
        semobridge_logits,
        semobridge_conv_images_logits,
        semobridge_conv_prompt_logits,
        params,
    ):
        """Blend the logits from CLIP and Semobridge."""

        if "smoothness" in params.keys():
            weighted_few_shot_divergence = (
                few_shot_divergence * params["smoothness"]
            ).exp()
            soft_few_shot_labels = (
                self.features.few_shot_labels * weighted_few_shot_divergence
            )

        if "alpha" in params.keys():
            clip_logits_exp = (
                (-1) * (params["alpha"] - params["alpha"] * clip_logits)
            ).exp()
        if "beta" in params.keys():
            semobridge_logits_exp = (
                (-1) * (params["beta"] - params["beta"] * semobridge_logits)
            ).exp() @ soft_few_shot_labels
        if "gamma" in params.keys():
            semobridge_conv_images_logits_exp = (
                (-1)
                * (params["gamma"] - params["gamma"] * semobridge_conv_images_logits)
            ).exp() @ soft_few_shot_labels
        if "delta" in params.keys():
            semobridge_conv_prompt_logits_exp = (
                (-1)
                * (params["delta"] - params["delta"] * semobridge_conv_prompt_logits)
            ).exp() @ soft_few_shot_labels

        # Blend logits batch-wise
        blended_logits = torch.zeros(
            clip_logits.shape[0],
            clip_logits.shape[1],
            device=self.device,
            dtype=self.features.dtype,
        )
        if "lambda1" in params.keys():
            blended_logits += params["lambda1"] * clip_logits_exp
        if "lambda2" in params.keys():
            blended_logits += params["lambda2"] * semobridge_logits_exp
        if "lambda3" in params.keys():
            blended_logits += params["lambda3"] * semobridge_conv_images_logits_exp
        if "lambda4" in params.keys():
            blended_logits += params["lambda4"] * semobridge_conv_prompt_logits_exp

        return blended_logits

    def run_hp_search(self):
        pass

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def convert_to_ood(self):
        cfg = self.cfg

        # If the dataset is imagenet-a, now we have to set the classnames to the actual dataset classnames and only use the corresponding final prompts and class biases
        if cfg.DATASET.NAME in OOD_DATASETS:
            print("Setting classnames to actual dataset classnames for OOD datasets")
            self.classnames = [
                name.replace("_", " ") for name in self.dm.dataset.classnames
            ]
            self.num_classes = len(self.classnames)

            new_class_bias = torch.zeros(
                self.num_classes,
                self.features.text_projection.shape[0],
                dtype=self.features.dtype,
            )
            for i, classname in enumerate(self.classnames):
                if classname not in self.features.imagenet_classnames:
                    raise ValueError(
                        f"Class {classname} not found in ImageNet classnames"
                    )

                class_id = self.features.imagenet_classnames.index(classname)
                if self.cfg.TRAINER.SEMOBRIDGE.CSB:
                    new_class_bias[i] = self.semobridge.class_bias[class_id]

            if self.cfg.TRAINER.SEMOBRIDGE.CSB:
                new_class_bias = new_class_bias.to(self.device)
                self.semobridge.class_bias = torch.nn.Parameter(new_class_bias)

    def load_model(self, directory, model_file=None, epoch=None):
        cfg = self.cfg

        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        if model_file is None:
            model_file = "model-best.pth.tar"
        else:
            model_file = model_file + ".pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False

            self._models[name].load_state_dict(state_dict, strict=False)

            self.convert_to_ood()
