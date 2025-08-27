import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
from dassl.data.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from utils import generate_train_split_slug


@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):

    dataset_dir = "oxford_pets"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            trainval = self.read_data(split_file="trainval.txt")
            test = self.read_data(split_file="test.txt")
            train, val = self.split_trainval(trainval)
            self.save_split(train, val, test, self.split_path, self.image_dir)

        train, val = self.load_fewshot_dataset(
            self=self,
            cfg=cfg,
            split_fewshot_dir=self.split_fewshot_dir,
            train=train,
            val=val,
        )

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f"Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val")
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)

        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}

        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname,
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output

    @staticmethod
    def load_fewshot_dataset(self, cfg, split_fewshot_dir, train, val):
        num_shots = cfg.DATASET.NUM_SHOTS

        augment_cfg = cfg.INPUT.FEW_SHOT_AUGMENTATION.get(str(num_shots), None)
        if augment_cfg is None:
            raise ValueError(
                f"Few-shot augmentation configuration for {num_shots} shots not found in cfg.INPUT.FEW_SHOT_AUGMENTATION."
            )
        augment_epochs = augment_cfg.AUGMENT_EPOCHS

        if num_shots >= 1:
            self.variant_name = f"{generate_train_split_slug(cfg)}.pkl"
            preprocessed = os.path.join(split_fewshot_dir, self.variant_name)

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = OxfordPets.generate_fewshot_dataset(
                    self,
                    train,
                    cfg=cfg,
                    num_shots=num_shots,
                    augment_epochs=augment_epochs,
                    split_fewshot_dir=split_fewshot_dir,
                )
                # We do not want to generate few-shot val set
                # if val is not None:
                #     val = OxfordPets.generate_fewshot_dataset(
                #         self,
                #         val,
                #         cfg=cfg,
                #         num_shots=min(num_shots, 4),
                #         augment_epochs=0,
                #         split_fewshot_dir=split_fewshot_dir,
                #     )
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        return train, val

    @staticmethod
    def generate_fewshot_dataset(
        self,
        *data_sources,
        cfg,
        num_shots=-1,
        augment_epochs=0,
        split_fewshot_dir,
        repeat=False,
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            augment_epochs (int): number of epochs to augment the shots.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        augment_cfg = cfg.INPUT.FEW_SHOT_AUGMENTATION.get(str(num_shots), None)
        if augment_cfg is None:
            raise ValueError(
                f"Few-shot augmentation configuration for {num_shots} shots not found in cfg.INPUT.FEW_SHOT_AUGMENTATION."
            )

        print(f"Creating a {num_shots}-shot augmented {augment_epochs} epochs dataset")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                    print(f"label: {label}, img 0: {sampled_items[0].impath}")
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            # Augment the dataset
            if augment_epochs > 0:
                # Create a directory for augmented images
                augment_dir = os.path.join(
                    split_fewshot_dir,
                    "augments",
                    self.variant_name,
                )
                mkdir_if_missing(augment_dir)

                INTERPOLATION_MODES = {
                    "bilinear": InterpolationMode.BILINEAR,
                    "bicubic": InterpolationMode.BICUBIC,
                    "nearest": InterpolationMode.NEAREST,
                }
                interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
                transforms_list = []
                if "random_resized_crop" in augment_cfg.AUGMENT_TRANSFORMS:
                    transforms_list.append(
                        transforms.RandomResizedCrop(
                            size=cfg.INPUT.SIZE[0],
                            scale=augment_cfg.RRCROP_SCALE,
                            ratio=(1.0, 1.0),
                            interpolation=interp_mode,
                        )
                    )
                if "random_flip" in augment_cfg.AUGMENT_TRANSFORMS:
                    transforms_list.append(transforms.RandomHorizontalFlip())
                if "horizontal_flip" in augment_cfg.AUGMENT_TRANSFORMS:
                    transforms_list.append(transforms.RandomHorizontalFlip(1.0))
                if "color_jitter" in augment_cfg.AUGMENT_TRANSFORMS:
                    transforms_list.append(
                        transforms.ColorJitter(
                            brightness=augment_cfg.COLORJITTER_B,
                            contrast=augment_cfg.COLORJITTER_C,
                            saturation=augment_cfg.COLORJITTER_S,
                            hue=augment_cfg.COLORJITTER_H,
                        )
                    )
                augment_transform = transforms.Compose(transforms_list)

                dataset_orig = dataset.copy()
                for augment_idx in range(augment_epochs):

                    print(
                        "Augment Epoch: {:} / {:}".format(augment_idx, augment_epochs)
                    )
                    for item in dataset_orig:
                        new_impath = os.path.join(
                            augment_dir,
                            f"{os.path.basename(item.impath)}_augment_{augment_idx}{os.path.splitext(item.impath)[1]}",
                        )
                        # Check if the augmented image does not exist
                        if not os.path.exists(new_impath):
                            # Load image
                            image = Image.open(item.impath).convert("RGB")
                            # Apply augmentations
                            image = augment_transform(image)
                            # Save augmented image
                            image.save(new_impath)

                        item_new = Datum(
                            impath=new_impath,
                            label=item.label,
                            classname=item.classname,
                        )
                        dataset.append(item_new)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output
