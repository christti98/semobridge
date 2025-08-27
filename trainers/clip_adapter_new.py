import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from ..imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from .zsclip import CUSTOM_TEMPLATES


_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model, projection_dims):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.device = "cuda"
        self.projection_dims = projection_dims

        self.texts = self.load_texts(cfg)
        self.text_features = self.load_class_text_prompts(self.texts)

    def load_texts(self, cfg):
        texts_cfg = cfg.TRAINER.CLIP_ADAPTER.TEXTS
        
        if texts_cfg == "classname":
            texts = [[c] for c in self.classnames]
        elif texts_cfg == "aphotoofa":
            texts = [[f"a photo of a {c}"] for c in self.classnames]
        elif texts_cfg == "clip":
            temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            texts = [[temp.format(c)] for c in self.classnames]
        elif texts_cfg == "clip_ensemble":
            templates = IMAGENET_TEMPLATES_SELECT
            # add custom-made prompt
            if cfg.DATASET.NAME != "ImageNet":
                templates.append(CUSTOM_TEMPLATES[cfg.DATASET.NAME])

            self.texts = [[temp.format(c) for temp in templates] for c in self.classnames]
        elif texts_cfg == "cupl_base":
            json_path = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.DIRECTORY, "cupl_base.json") 
            with open(json_path, "r") as f:
                json_file = json.load(f)
                # Convert keys to lowercase
                json_file = {k.lower(): v for k, v in json_file.items()}
            texts = []
            for i in range(self.num_classes):
                texts.append(json_file[self.classnames[i].lower()])
        elif texts_cfg == "cupl_full":
            json_path = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.DIRECTORY, "cupl_full.json")
            with open(json_path, "r") as f:
                json_file = json.load(f)
                # Convert keys to lowercase
                json_file = {k.lower(): v for k, v in json_file.items()}
            texts = []
            for i in range(self.num_classes):
                texts.append(json_file[self.classnames[i].lower()])
        else:
            raise ValueError(f"Unknown text type: {texts_cfg}")

        return texts
    
    def load_class_text_prompts(self, templates):
        text_projected = torch.empty(
            self.num_classes,
            self.projection_dims,
            device=self.device,
            dtype=self.dtype,
        )

        for class_id in range(self.num_classes):
            # Tokenize the text
            class_text_tokens = clip.tokenize(templates[class_id]).to(self.device)

            # Embed the text
            temp = self.encode_text(
                class_text_tokens, projection=True, pooling=True
            )
            text_projected[class_id] = temp.mean(dim=0)

        return text_projected

    def forward(self):
        return self.text_features


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.projection_dims = clip_model.text_projection.shape[1]
        self.text_encoder = TextEncoder(cfg, classnames, clip_model, self.projection_dims)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        

        self.adapter = Adapter(self.projection_dims, 4).to(clip_model.dtype)


            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CLIP_Adapter(TrainerX):
    """ CLIP-Adapter """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        
        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        

        self.register_model('clip_adapter', self.model.adapter, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            
            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
