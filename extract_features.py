import os.path as osp

import torch
from dassl.data.data_manager import build_data_loader, build_transform

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from tqdm import tqdm
from imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
import os
import json
from utils import generate_train_split_slug

import torch.nn.functional as F

from custom_clip import CustomCLIP
from datasets.imagenet import ImageNet

OOD_DATASETS = ["ImageNetA", "ImageNetR", "ImageNetSketch", "ImageNetV2"]

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class FeatureExtractor:
    def __init__(self, cfg, dm, device):
        self.cfg = cfg
        self.dm = dm
        self.device = device

    # From dassl/engine/trainer.py/SimpleTrainer
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    @torch.no_grad()
    def load_clip_if_not_loaded(self):
        if hasattr(self, "model") and self.model is not None:
            return

        # Load CLIP model
        print(f"Loading CLIP ({self.cfg.MODEL.BACKBONE.NAME})...")

        url = clip._MODELS[self.cfg.MODEL.BACKBONE.NAME]
        model_path = clip._download(url)

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        clip_model = clip.build_model(state_dict or model.state_dict())
        # self.clip_model.eval()
        if (
            self.cfg.TRAINER.SEMOBRIDGE.PREC == "fp32"
            or self.cfg.TRAINER.SEMOBRIDGE.PREC == "amp"
        ):
            # CLIP's default precision is fp16
            clip_model = clip_model.float()

        self.model = CustomCLIP(self.cfg.MODEL.BACKBONE.NAME, clip_model)
        self.model.eval()
        self.model.to(self.device)

        self.projection_dims = self.model.projection_dims
        self.text_encoding_dims = self.model.text_encoding_dims
        self.dtype = self.model.dtype

    @torch.no_grad()
    def unload_clip(self):
        """Unload the CLIP model to free up memory."""
        if hasattr(self, "model"):
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            print("CLIP model unloaded.")
        else:
            print("No CLIP model to unload.")

    @torch.no_grad()
    def load_class_text_prompts(self, text_types_override=None):
        text_types = self.cfg.TRAINER.SEMOBRIDGE.TEXTS.split(",")
        if text_types_override is not None:
            text_types = text_types_override.split(",")

        print(text_types)

        texts = []
        for i in range(self.num_classes):
            texts.append([])

        for text_type in text_types:
            if text_type == "classname":
                for i in range(self.num_classes):
                    texts[i].extend([self.classnames[i]])
            elif text_type == "aphotoofa":
                for i in range(self.num_classes):
                    texts[i].extend([f"a photo of a {self.classnames[i]}"])
            elif text_type == "clip":
                temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]

                for i in range(self.num_classes):
                    texts[i].extend([temp.format(self.classnames[i])])
            elif text_type == "clip_ensemble":
                templates = IMAGENET_TEMPLATES_SELECT
                # add custom-made prompt
                if self.cfg.DATASET.NAME != "ImageNet":
                    for i in range(self.num_classes):
                        texts[i].append(
                            CUSTOM_TEMPLATES[self.cfg.DATASET.NAME].format(
                                self.classnames[i]
                            )
                        )
                # else:

                no_imagenet_ensemble = [
                    "DescribableTextures",
                    "EuroSAT",
                    "OxfordFlowers",
                ]
                if self.cfg.DATASET.NAME not in no_imagenet_ensemble:
                    for i in range(self.num_classes):
                        texts[i].extend(
                            [temp.format(self.classnames[i]) for temp in templates]
                        )
            elif text_type.startswith("cupl_"):
                json_path = os.path.join(
                    "TEXTS",
                    self.cfg.DATASET.DIRECTORY,
                    f"{text_type}.json",
                )
                with open(json_path, "r") as f:
                    json_file = json.load(f)
                    # Convert keys to lowercase
                    json_file = {k.lower(): v for k, v in json_file.items()}
                for i in range(self.num_classes):
                    texts[i].extend(json_file[self.classnames[i].lower()])
            else:
                raise ValueError(f"Unknown text type: {text_type}")

        # Remove duplicates
        # for i in range(self.num_classes):
        #     texts[i] = list(set(texts[i]))
        # print(texts[0])

        return texts

    @torch.no_grad()
    def embed_class_text_prompts(self, texts):
        text_projected = torch.empty(
            self.num_classes,
            # len(templates[0]),
            self.projection_dims,
            device=self.device,
            dtype=self.dtype,
        )
        text_encoded = torch.empty(
            self.num_classes,
            self.text_encoding_dims,
            device=self.device,
            dtype=self.dtype,
        )
        text_projected_normed = torch.empty(
            self.num_classes,
            self.projection_dims,
            device=self.device,
            dtype=self.dtype,
        )
        text_encoded_normed = torch.empty(
            self.num_classes,
            self.text_encoding_dims,
            device=self.device,
            dtype=self.dtype,
        )

        # text_encoded_not_pooled = torch.empty(
        #     self.num_classes,
        #     77,
        #     self.text_encoding_dims,
        #     device=self.device,
        #     dtype=self.dtype,
        # )

        # norms = []

        for class_id in range(self.num_classes):
            if len(texts[class_id]) == 0:
                raise ValueError(
                    f"Empty template for class {class_id}: {self.classnames[class_id]}"
                )

            # Tokenize the text
            class_text_tokens = clip.tokenize(texts[class_id], truncate=True).to(
                self.device
            )

            # Embed the text
            temp = self.model.encode_text_custom(
                class_text_tokens, projection=True, pooling=True
            )
            # text_projected[class_id] = temp
            text_projected[class_id] = temp.mean(dim=0)

            temp_normed = temp / temp.norm(dim=-1, keepdim=True)
            text_projected_normed[class_id] = temp_normed.mean(dim=0)
            text_projected_normed[class_id] /= text_projected_normed[class_id].norm()

            # temp = self.encode_text_custom(
            #     class_text_tokens, projection=False, pooling=False
            # )
            # text_encoded_not_pooled[class_id] = temp.mean(dim=0)

            temp = self.model.encode_text_custom(
                class_text_tokens, projection=False, pooling=True
            )
            text_encoded[class_id] = temp.mean(dim=0)

            temp_normed = temp / temp.norm(dim=-1, keepdim=True)
            text_encoded_normed[class_id] = temp_normed.mean(dim=0)
            text_encoded_normed[class_id] /= text_encoded_normed[class_id].norm()

            # temp = self.encode_text_custom(
            #     class_text_tokens, projection=False, pooling=False
            # )
            # text_encoded_not_pooled[class_id] = temp.mean(dim=0)

        # text_projected_normed = F.normalize(text_projected, dim=-1)
        # text_encoded_normed = F.normalize(text_encoded, dim=-1)

        # Calculate average norm
        # avg_norm = sum(norms) / len(norms)
        # print(f"Average text norm: {avg_norm:.4f}")

        return text_projected, text_projected_normed, text_encoded, text_encoded_normed
        # return text_projected, text_encoded, text_encoded_not_pooled
        # text_encoded,
        # text_encoded_normed,
        # text_encoded_not_pooled,

    @torch.no_grad()
    def load_fewshot_images(self):
        tfm = build_transform(self.cfg, is_train=False)

        # Convert the image embeddings to text embeddings for the validation set
        dataloader_temp = build_data_loader(
            self.cfg,
            # sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
            sampler_type="SequentialSampler",  # Force sequential
            data_source=self.dm.dataset.train_x,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            tfm=tfm,
            is_train=False,
        )

        image_embeds = torch.empty(
            self.num_classes,
            self.num_shots,
            self.model.projection_dims,
            device=self.device,
            dtype=self.model.dtype,
        )
        added_shots = torch.zeros(
            self.num_classes, device=self.device, dtype=torch.long
        )  # Track the number of shots added for each class
        for batch in tqdm(dataloader_temp):
            images = batch["img"].to(self.device).type(self.model.dtype)
            labels = batch["label"].to(self.device)

            # Encode the images
            embeds = self.model.encode_image(images)
            for index in range(batch["img"].shape[0]):
                class_id = labels[index]
                shot_id = added_shots[class_id]
                image_embeds[class_id][shot_id] = embeds[index]
                added_shots[class_id] += 1

        return image_embeds

    @torch.no_grad()
    def load_split(self, split):
        """Load a specific split of the dataset."""
        if split == "val" and self.dm.val_loader is not None:
            data_loader = self.dm.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.dm.test_loader

        num_batches = len(data_loader)
        test_embeds = []
        test_labels = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            images_projected = self.model.encode_image(
                input.type(self.model.dtype)
            )  # Image encoding with projection
            # images_projected_normed = F.normalize(images_projected, dim=-1)
            test_embeds.append(images_projected)
            test_labels.append(label)

        # Save the preprocessed image embeddings
        test_embeds = torch.cat(test_embeds, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        return test_embeds, test_labels

    @torch.no_grad()
    def extract_features(self, output_dir="preprocessed"):
        cfg = self.cfg

        if (
            self.cfg.TRAINER.SEMOBRIDGE.PREC == "fp32"
            or self.cfg.TRAINER.SEMOBRIDGE.PREC == "amp"
        ):
            self.dtype = torch.float32
        elif self.cfg.TRAINER.SEMOBRIDGE.PREC == "fp16":
            self.dtype = torch.float16
        else:
            raise ValueError(f"Unknown precision: {self.cfg.TRAINER.SEMOBRIDGE.PREC}")

        ### PROCESS DATASET
        print(f"Extracting features for dataset: {cfg.DATASET.NAME}")

        # Create output directory if it doesn't exist
        backbone_dir = osp.join(
            output_dir,
            f"{cfg.MODEL.BACKBONE.NAME.replace("/", "")}_{cfg.TRAINER.SEMOBRIDGE.PREC}",
        )
        if not osp.exists(backbone_dir):
            os.makedirs(backbone_dir)
        dataset_dir = osp.join(backbone_dir, cfg.DATASET.DIRECTORY)
        if cfg.DATASET.SUBSAMPLE_CLASSES != "all":
            dataset_dir = osp.join(
                dataset_dir, f"subsample_{cfg.DATASET.SUBSAMPLE_CLASSES}"
            )

        if not osp.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # If the dataset is OOD, we have to set the classnames to imagenet temporarily (to load final prompts and class bias)
        if cfg.DATASET.NAME in OOD_DATASETS:
            self.imagenet_classnames = ImageNet.read_classnames(
                f"{cfg.DATASET.ROOT}/imagenet/classnames.txt"
            )
            self.imagenet_classnames = list(self.imagenet_classnames.values())
            self.imagenet_classnames = [
                name.replace("_", " ") for name in self.imagenet_classnames
            ]

        self.classnames = self.dm.dataset.classnames
        self.classnames = [name.replace("_", " ") for name in self.classnames]
        self.num_classes = len(self.classnames)

        # Calculate real shots per class with cfg.DATASET.NUM_SHOTS and cfg.INPUT.FEW_SHOT_AUGMENTATION[<num_shots>]
        self.num_shots = self.cfg.DATASET.NUM_SHOTS * (
            self.cfg.INPUT.FEW_SHOT_AUGMENTATION[
                str(self.cfg.DATASET.NUM_SHOTS)
            ].AUGMENT_EPOCHS
            + 1
        )

        ### TEXT PROJECTION FROM CLIP (SD-IPC initialization) and LOGIT SCALE
        save_path = os.path.join(
            backbone_dir,
            f"clip_data.pt",
        )
        if not osp.exists(save_path):
            self.load_clip_if_not_loaded()

            print(f"Extracting CLIP logit scale, text projection and pseudo-inverse...")
            self.logit_scale = self.model.logit_scale
            self.text_projection = self.model.clip_model.text_projection.data
            torch.save([self.logit_scale, self.text_projection], save_path)
            print(f"Saved CLIP logit scale and text projection {save_path}")
        else:
            print(f"Loading CLIP logit scale and text projection from {save_path}")
            self.logit_scale, self.text_projection = torch.load(save_path)

        ### TEXT EMBEDDINGS
        save_path = os.path.join(
            dataset_dir,
            f"{self.cfg.TRAINER.SEMOBRIDGE.TEXTS}.pt",
        )
        if not osp.exists(save_path):
            self.load_clip_if_not_loaded()

            print(
                f"Generating {self.cfg.TRAINER.SEMOBRIDGE.TEXTS} for {self.num_classes} classes..."
            )
            texts = self.load_class_text_prompts()
            text_projected, text_projected_normed, text_encoded, text_encoded_normed = (
                self.embed_class_text_prompts(texts)
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
            print(f"Saved text embeddings to {save_path}")
        else:
            print(f"Loading {self.cfg.TRAINER.SEMOBRIDGE.TEXTS} from {save_path}")
            text_projected, text_projected_normed, text_encoded, text_encoded_normed = (
                torch.load(save_path)
            )
        self.text_projected = text_projected
        self.text_encoded = text_encoded
        self.text_projected_normed = text_projected_normed
        self.text_encoded_normed = text_encoded_normed

        ### FEW-SHOT IMAGE EMBEDDINGS (shotwise, [num_classes, num_shots, projection_dims])
        split = "train"
        save_path = os.path.join(
            dataset_dir,
            f"{split}_image_embeddings_{generate_train_split_slug(self.cfg)}.pt",
        )
        if not osp.exists(save_path) and self.cfg.DATASET.NAME not in OOD_DATASETS:
            self.load_clip_if_not_loaded()

            print(f"Generating few-shot image embeddings...")
            self.few_shot_embeds = self.load_fewshot_images()
            torch.save(self.few_shot_embeds, save_path)
            print(f"Saved {split} image embeddings to {save_path}")
        else:
            if self.cfg.DATASET.NAME in OOD_DATASETS:
                print(f"Using original ImageNet for few-shot images for OOD test.")
                save_path = os.path.join(
                    backbone_dir,
                    "imagenet",
                    f"{split}_image_embeddings_{generate_train_split_slug(self.cfg)}.pt",
                )

            print(f"Loading few-shot image embeddings from {save_path}")
            self.few_shot_embeds = torch.load(save_path)

        if self.cfg.DATASET.NAME in OOD_DATASETS:
            # Only keep the classes we need
            self.few_shot_embeds_new = torch.empty(
                self.num_classes,
                self.num_shots,
                self.few_shot_embeds.shape[-1],
                device=self.device,
                dtype=self.few_shot_embeds.dtype,
            )

            for i, classname in enumerate(self.classnames):
                if classname not in self.imagenet_classnames:
                    raise ValueError(
                        f"Class {classname} not found in ImageNet classnames"
                    )

                class_id = self.imagenet_classnames.index(classname)
                self.few_shot_embeds_new[i] = self.few_shot_embeds[class_id]

            self.few_shot_embeds = self.few_shot_embeds_new

        self.few_shot_embeds_flat = self.few_shot_embeds.flatten(start_dim=0, end_dim=1)
        self.few_shot_embeds_flat_normed = F.normalize(
            self.few_shot_embeds_flat, dim=-1
        )
        self.few_shot_embeds_mean = self.few_shot_embeds.mean(dim=1)
        self.few_shot_embeds_mean_normed = F.normalize(
            self.few_shot_embeds_mean, dim=-1
        )
        # Make [C*K, C] one-hot labels for the few shot set
        self.few_shot_labels = torch.zeros(
            self.num_classes,
            self.num_classes,
            device=self.device,
            dtype=torch.float32,
        )
        for i in range(self.num_classes):
            self.few_shot_labels[i][i] = 1.0

        ### TEST IMAGE EMBEDDINGS AND LABELS
        split = "test"
        save_path = os.path.join(
            dataset_dir,
            f"{split}_image_embeddings.pt",
        )
        if not osp.exists(save_path):
            self.load_clip_if_not_loaded()

            print(f"Generating {split} image embeddings...")
            self.test_embeds, self.test_labels = self.load_split(split)
            torch.save([self.test_embeds, self.test_labels], save_path)
            print(f"Saved {split} image embeddings and labels to {save_path}")
        else:
            print(f"Loading {split} image embeddings from {save_path}")
            self.test_embeds, self.test_labels = torch.load(save_path)
        self.test_embeds_normed = F.normalize(self.test_embeds, dim=-1)

        ### VAL IMAGE EMBEDDINGS AND LABELS
        split = "val"
        if self.dm.val_loader is not None:
            save_path = os.path.join(
                dataset_dir,
                f"{split}_image_embeddings.pt",
            )
            if not osp.exists(save_path):
                self.load_clip_if_not_loaded()

                print(f"Generating {split} image embeddings...")
                self.val_embeds, self.val_labels = self.load_split(split)
                torch.save([self.val_embeds, self.val_labels], save_path)
                print(f"Saved {split} image embeddings and labels to {save_path}")
            else:
                print(f"Loading {split} image embeddings from {save_path}")
                self.val_embeds, self.val_labels = torch.load(save_path)
        else:
            print(f"No validation set, using test as val.")
            self.val_embeds = (self.test_embeds,)
            self.val_labels = self.test_labels
        self.val_embeds_normed = F.normalize(self.val_embeds, dim=-1)

        ### CALCULATE CLIP LOGITS
        save_path = os.path.join(
            dataset_dir,
            f"val_clip_logits_{self.cfg.TRAINER.SEMOBRIDGE.TEXTS}.pt",
        )
        if not osp.exists(save_path):
            self.load_clip_if_not_loaded()

            print(
                f"Calculating CLIP logits for {split} using {self.cfg.TRAINER.SEMOBRIDGE.TEXTS}..."
            )
            self.val_clip_logits = self.val_embeds_normed @ self.text_projected_normed.T
            torch.save(self.val_clip_logits, save_path)
            print(f"Saved {split} CLIP logits to {save_path}")
        else:
            print(f"Loading {split} CLIP logits from {save_path}")
            self.val_clip_logits = torch.load(save_path)

        save_path = os.path.join(
            dataset_dir,
            f"test_clip_logits_{self.cfg.TRAINER.SEMOBRIDGE.TEXTS}.pt",
        )
        if not osp.exists(save_path):
            self.load_clip_if_not_loaded()

            print(
                f"Calculating CLIP logits for {split} using {self.cfg.TRAINER.SEMOBRIDGE.TEXTS}..."
            )
            self.test_clip_logits = (
                self.test_embeds_normed @ self.text_projected_normed.T
            )
            torch.save(self.test_clip_logits, save_path)
            print(f"Saved {split} CLIP logits to {save_path}")
        else:
            print(f"Loading {split} CLIP logits from {save_path}")
            self.test_clip_logits = torch.load(save_path)

        self.valimg_fewshot_logits = (
            self.val_embeds_normed @ self.few_shot_embeds_mean_normed.T
        )
        self.testimg_fewshot_logits = (
            self.test_embeds_normed @ self.few_shot_embeds_mean_normed.T
        )

        self.few_shot_clip_logits = (
            self.few_shot_embeds_mean_normed @ self.text_projected_normed.T
        )
        self.few_shot_clip_logits = self.few_shot_clip_logits.softmax(dim=1)
        self.few_shot_divergence = torch.sum(
            self.few_shot_labels
            * torch.log2(
                (self.few_shot_labels + 1e-6) / (self.few_shot_clip_logits + 1e-6)
            ),
            dim=1,
        )[:, None]

        # Unload the CLIP model to free up memory
        self.unload_clip()

    def clear_memory(self):
        # Clear everything except val and test labels
        del self.text_projection
        del self.text_projected
        del self.text_encoded
        #del self.text_projected_normed
        del self.text_encoded_normed
        del self.few_shot_embeds
        del self.few_shot_embeds_flat
        del self.few_shot_embeds_flat_normed
        del self.few_shot_embeds_mean
        # del self.few_shot_embeds_mean_normed
        # del self.few_shot_labels
        del self.val_embeds
        del self.val_embeds_normed
        # del self.val_clip_logits
        # del self.test_embeds
        # del self.test_embeds_normed
        # del self.test_clip_logits
        del self.valimg_fewshot_logits
        del self.testimg_fewshot_logits

        # Clear CUDA cache
        torch.cuda.empty_cache()
        # Run garbage collection
        import gc

        gc.collect()
