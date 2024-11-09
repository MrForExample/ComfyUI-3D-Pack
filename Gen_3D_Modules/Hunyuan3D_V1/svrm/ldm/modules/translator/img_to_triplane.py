import math
import torch
import torch.nn as nn
from ..attention import ImgToTriplaneTransformer
import math
from einops import rearrange


class ImgToTriplaneModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        pos_emb_size=32,
        pos_emb_dim=1024,
        cam_cond_dim=20,
        n_heads=16,
        d_head=64,
        depth=16,
        context_dim=768,
        triplane_dim=80,
        upsample_time=1,
        use_fp16=False,
        use_bf16=True,
    ):
        super().__init__()

        self.pos_emb_size = pos_emb_size
        self.pos_emb_dim  = pos_emb_dim

        # init embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, 3 * pos_emb_size * pos_emb_size, pos_emb_dim))
        # TODO initialize pos_emb with a Gaussian random of zero-mean and std of 1/sqrt(1024).

        # build image to triplane decoder
        self.img_to_triplane_decoder = ImgToTriplaneTransformer(
            query_dim=pos_emb_dim, n_heads=n_heads,
            d_head=d_head, depth=depth, context_dim=context_dim,
            triplane_size=pos_emb_size, 
        )

        self.is_conv_upsampler = False
        # build upsampler
        self.triplane_dim = triplane_dim
        if self.is_conv_upsampler:
            upsamplers = []
            for i in range(upsample_time):
                if i == 0:
                    upsampler = nn.ConvTranspose2d(in_channels=pos_emb_dim, out_channels=triplane_dim,
                                            kernel_size=2, stride=2,
                                            padding=0, output_padding=0)
                    upsamplers.append(upsampler)
                else:
                    upsampler = nn.ConvTranspose2d(in_channels=triplane_dim, out_channels=triplane_dim,
                                            kernel_size=2, stride=2,
                                            padding=0, output_padding=0)
                    upsamplers.append(upsampler)
            if upsamplers:
                self.upsampler = nn.Sequential(*upsamplers)
            else:
                self.upsampler = nn.Conv2d(in_channels=pos_emb_dim, out_channels=triplane_dim,
                                            kernel_size=3, stride=1, padding=1)
        else:
            self.upsample_ratio = 4
            self.upsampler = nn.Linear(in_features=pos_emb_dim, out_features=triplane_dim*(self.upsample_ratio**2))
        


    def forward(self, x, cam_cond=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        B = x.shape[0]
        h = self.pos_emb.expand(B, -1, -1)
        context = x

        h = self.img_to_triplane_decoder(h, context=context)

        h = h.view(B * 3, self.pos_emb_size, self.pos_emb_size, self.pos_emb_dim)
        if self.is_conv_upsampler:
            h = rearrange(h, 'b h w c -> b c h w')
            h = self.upsampler(h)
            h = rearrange(h, '(b d) c h w-> b d c h w', d=3)
            h = h.type(x.dtype)
            return h 
        else:
            h = self.upsampler(h) #[b, h, w, triplane_dim*4]
            b, height, width, _ = h.shape
            h = h.view(b, height, width, self.triplane_dim, self.upsample_ratio, self.upsample_ratio) #[b, h, w, triplane_dim, 2, 2]
            h = h.permute(0,3,1,4,2,5).contiguous() #[b, triplane_dim, h, 2, w, 2]
            h = h.view(b, self.triplane_dim, height*self.upsample_ratio, width*self.upsample_ratio)
            h = rearrange(h, '(b d) c h w-> b d c h w', d=3)
            h = h.type(x.dtype)
            return h 
