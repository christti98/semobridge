import torch
import torch.nn as nn


class CustomCLIP(nn.Module):
    def __init__(self, backbone_name, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.is_resnet = backbone_name.startswith("RN")
        self.vision_encoding_dims = (
            self.clip_model.visual.attnpool.c_proj.in_features
            if self.is_resnet
            else self.clip_model.visual.proj.shape[0]
        )
        self.text_encoding_dims = self.clip_model.text_projection.shape[0]
        self.projection_dims = self.clip_model.text_projection.shape[1]

        # ViT-B
        # self.text_encoding_dims = 512
        # self.projection_dims = 512
        # RN50
        # self.text_encoding_dims = 512
        # self.projection_dims = 1024

        self.dtype = clip_model.dtype

        self.text_projection = clip_model.text_projection.data

        # self.logit_scale = clip_model.logit_scale
        self.logit_scale = clip_model.logit_scale.detach().exp()

    def encode_image(self, image):
        return self.clip_model.encode_image(image)

    def encode_text(self, text):
        return self.clip_model.encode_text(text)

    def encode_text_custom(self, text, projection=True, pooling=True):
        x = self.clip_model.token_embedding(text).type(
            self.dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        if pooling:
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        if projection:
            x = x @ self.clip_model.text_projection

        return x

    def forward(self, image):
        raise NotImplementedError("SeMoBridge does not support forward method.")
