import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

import numpy as np
import open_clip
from PIL import Image
from ...util import default, count_params


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):
    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key="class", ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0.0 and not disable_dropout:
            mask = 1.0 - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * torch.ones_like(c) * (self.n_classes - 1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = (
            self.n_classes - 1
        )  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=self.layer == "hidden"
        )
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder, nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        ip_mode=None
    ):
        """_summary_

        Args:
            ip_mode (str, optional): what is the image promcessing mode. Defaults to None.

        """
        super().__init__()
        assert layer in self.LAYERS
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version
        )
        if ip_mode is None:
            del model.visual
            
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.max_length = max_length
        self.ip_mode = ip_mode
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z
    
    def forward_image(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        if isinstance(pil_image, torch.Tensor):
            pil_image = pil_image.cpu().numpy()
        if isinstance(pil_image, np.ndarray):
            if pil_image.ndim == 3:
                pil_image = pil_image[None, :, :, :]
            pil_image = [Image.fromarray(x) for x in pil_image]

        images = []
        for image in pil_image:
            images.append(self.preprocess(image).to(self.device))

        image = torch.stack(images, 0) # to [b, 3, h, w]
        if self.ip_mode == "global":
            image_features = self.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        elif "local" in self.ip_mode:
            image_features = self.encode_image_with_transformer(image)

        return image_features # b, l
    
    def encode_image_with_transformer(self, x):
        visual = self.model.visual
        x = visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [visual.class_embedding.to(x.dtype) + \
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        # x = visual.patch_dropout(x) 
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        hidden = self.image_transformer_forward(x)
        x = hidden[-2].permute(1, 0, 2)  # LND -> NLD
        return x
    
    def image_transformer_forward(self, x):
        encoder_states = ()
        trans = self.model.visual.transformer
        for r in trans.resblocks:
            if trans.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, None)
            else:
                x = r(x, attn_mask=None)
            encoder_states = encoder_states + (x, )
        return encoder_states
    
    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",
        t5_version="google/t5-v1_1-xl",
        device="cuda",
        clip_max_length=77,
        t5_max_length=77,
    ):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(
            clip_version, device, max_length=clip_max_length
        )
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder)*1.e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder)*1.e-6:.2f} M params."
        )

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]
