# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# --------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection


from diffusers.image_processor import PipelineImageInput
from diffusers.models import (
    AutoencoderKL,
    UNet2DConditionModel,
	ControlNetModel,
)
from diffusers.schedulers import (
	DDIMScheduler
)

from diffusers.utils import (
    BaseOutput,
    logging,
    replace_example_docstring,
)


from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.controlnet import StableDiffusionControlNetPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

import pdb



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
Examples:
```py
>>> import diffusers
>>> import torch

>>> pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
...     "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
... ).to("cuda")

>>> image = diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
>>> normals = pipe(image)

>>> vis = pipe.image_processor.visualize_normals(normals.prediction)
>>> vis[0].save("einstein_normals.png")
```
"""


@dataclass
class YosoOutput(BaseOutput):
    """
    Output class for Marigold monocular normals prediction pipeline.

    Args:
        prediction (`np.ndarray`, `torch.Tensor`):
            Predicted normals with values in the range [-1, 1]. The shape is always $numimages \times 3 \times height
            \times width$, regardless of whether the images were passed as a 4D array or a list.
        uncertainty (`None`, `np.ndarray`, `torch.Tensor`):
            Uncertainty maps computed from the ensemble, with values in the range [0, 1]. The shape is $numimages
            \times 1 \times height \times width$.
        latent (`None`, `torch.Tensor`):
            Latent features corresponding to the predictions, compatible with the `latents` argument of the pipeline.
            The shape is $numimages * numensemble \times 4 \times latentheight \times latentwidth$.
    """

    prediction: Union[np.ndarray, torch.Tensor]
    latent: Union[None, torch.Tensor]
    gaus_noise: Union[None, torch.Tensor]


class YosoPipeline(StableDiffusionControlNetPipeline):
    """ Pipeline for monocular normals estimation using the Marigold method: https://marigoldmonodepth.github.io.
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]



    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
        scheduler: Union[DDIMScheduler] = None,
        safety_checker: StableDiffusionSafetyChecker = None,
        text_encoder: CLIPTextModel = None,
        tokenizer: CLIPTokenizer = None,
        feature_extractor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = False,
        default_denoising_steps: Optional[int] = 1,
		default_processing_resolution: Optional[int] = 768,
        prompt="",
        empty_text_embedding=None,
        t_start: Optional[int] = 0,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            controlnet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
                )

        # TODO yoso ImageProcessor
        self.image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = MarigoldImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution
        self.empty_text_embedding = empty_text_embedding
        self.t_start= t_start # target_out latents

    def check_inputs(
        self,
        image: PipelineImageInput,
        num_inference_steps: int,
        ensemble_size: int,
        processing_resolution: int,
        resample_method_input: str,
        resample_method_output: str,
        batch_size: int,
        ensembling_kwargs: Optional[Dict[str, Any]],
        latents: Optional[torch.Tensor],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        output_type: str,
        output_uncertainty: bool,
    ) -> int:
        if num_inference_steps is None:
            raise ValueError("`num_inference_steps` is not specified and could not be resolved from the model config.")
        if num_inference_steps < 1:
            raise ValueError("`num_inference_steps` must be positive.")
        if ensemble_size < 1:
            raise ValueError("`ensemble_size` must be positive.")
        if ensemble_size == 2:
            logger.warning(
                "`ensemble_size` == 2 results are similar to no ensembling (1); "
                "consider increasing the value to at least 3."
            )
        if ensemble_size == 1 and output_uncertainty:
            raise ValueError(
                "Computing uncertainty by setting `output_uncertainty=True` also requires setting `ensemble_size` "
                "greater than 1."
            )
        if processing_resolution is None:
            raise ValueError(
                "`processing_resolution` is not specified and could not be resolved from the model config."
            )
        if processing_resolution < 0:
            raise ValueError(
                "`processing_resolution` must be non-negative: 0 for native resolution, or any positive value for "
                "downsampled processing."
            )
        if processing_resolution % self.vae_scale_factor != 0:
            raise ValueError(f"`processing_resolution` must be a multiple of {self.vae_scale_factor}.")
        if resample_method_input not in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
            raise ValueError(
                "`resample_method_input` takes string values compatible with PIL library: "
                "nearest, nearest-exact, bilinear, bicubic, area."
            )
        if resample_method_output not in ("nearest", "nearest-exact", "bilinear", "bicubic", "area"):
            raise ValueError(
                "`resample_method_output` takes string values compatible with PIL library: "
                "nearest, nearest-exact, bilinear, bicubic, area."
            )
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")
        if output_type not in ["pt", "np"]:
            raise ValueError("`output_type` must be one of `pt` or `np`.")
        if latents is not None and generator is not None:
            raise ValueError("`latents` and `generator` cannot be used together.")
        if ensembling_kwargs is not None:
            if not isinstance(ensembling_kwargs, dict):
                raise ValueError("`ensembling_kwargs` must be a dictionary.")
            if "reduction" in ensembling_kwargs and ensembling_kwargs["reduction"] not in ("closest", "mean"):
                raise ValueError("`ensembling_kwargs['reduction']` can be either `'closest'` or `'mean'`.")

        # image checks
        num_images = 0
        W, H = None, None
        if not isinstance(image, list):
            image = [image]
        for i, img in enumerate(image):
            if isinstance(img, np.ndarray) or torch.is_tensor(img):
                if img.ndim not in (2, 3, 4):
                    raise ValueError(f"`image[{i}]` has unsupported dimensions or shape: {img.shape}.")
                H_i, W_i = img.shape[-2:]
                N_i = 1
                if img.ndim == 4:
                    N_i = img.shape[0]
            elif isinstance(img, Image.Image):
                W_i, H_i = img.size
                N_i = 1
            else:
                raise ValueError(f"Unsupported `image[{i}]` type: {type(img)}.")
            if W is None:
                W, H = W_i, H_i
            elif (W, H) != (W_i, H_i):
                raise ValueError(
                    f"Input `image[{i}]` has incompatible dimensions {(W_i, H_i)} with the previous images {(W, H)}"
                )
            num_images += N_i

        # latents checks
        if latents is not None:
            if not torch.is_tensor(latents):
                raise ValueError("`latents` must be a torch.Tensor.")
            if latents.dim() != 4:
                raise ValueError(f"`latents` has unsupported dimensions or shape: {latents.shape}.")

            if processing_resolution > 0:
                max_orig = max(H, W)
                new_H = H * processing_resolution // max_orig
                new_W = W * processing_resolution // max_orig
                if new_H == 0 or new_W == 0:
                    raise ValueError(f"Extreme aspect ratio of the input image: [{W} x {H}]")
                W, H = new_W, new_H
            w = (W + self.vae_scale_factor - 1) // self.vae_scale_factor
            h = (H + self.vae_scale_factor - 1) // self.vae_scale_factor
            shape_expected = (num_images * ensemble_size, self.vae.config.latent_channels, h, w)

            if latents.shape != shape_expected:
                raise ValueError(f"`latents` has unexpected shape={latents.shape} expected={shape_expected}.")

        # generator checks
        if generator is not None:
            if isinstance(generator, list):
                if len(generator) != num_images * ensemble_size:
                    raise ValueError(
                        "The number of generators must match the total number of ensemble members for all input images."
                    )
                if not all(g.device.type == generator[0].device.type for g in generator):
                    raise ValueError("`generator` device placement is not consistent in the list.")
            elif not isinstance(generator, torch.Generator):
                raise ValueError(f"Unsupported generator type: {type(generator)}.")

        return num_images

    def progress_bar(self, iterable=None, total=None, desc=None, leave=True):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        progress_bar_config = dict(**self._progress_bar_config)
        progress_bar_config["desc"] = progress_bar_config.get("desc", desc)
        progress_bar_config["leave"] = progress_bar_config.get("leave", leave)
        if iterable is not None:
            return tqdm(iterable, **progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_resolution: Optional[int] = None,
        match_input_resolution: bool = True,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 1,
        ensembling_kwargs: Optional[Dict[str, Any]] = None,
        latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        output_type: str = "np",
        output_uncertainty: bool = False,
        output_latent: bool = False,
        skip_preprocess: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`),
                `List[torch.Tensor]`: An input image or images used as an input for the normals estimation task. For
                arrays and tensors, the expected value range is between `[0, 1]`. Passing a batch of images is possible
                by providing a four-dimensional array or a tensor. Additionally, a list of images of two- or
                three-dimensional arrays or tensors can be passed. In the latter case, all list elements must have the
                same width and height.
            num_inference_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, defaults to `1`):
                Number of ensemble predictions. Recommended values are 5 and higher for better precision, or 1 for
                faster inference.
            processing_resolution (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, matches the larger input image dimension. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_resolution (`bool`, *optional*, defaults to `True`):
                When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer
                side of the output will equal to `processing_resolution`.
            resample_method_input (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize input images to `processing_resolution`. The accepted values are:
                `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            resample_method_output (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize output predictions to match the input resolution. The accepted values
                are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size; only matters when setting `ensemble_size` or passing a tensor of images.
            ensembling_kwargs (`dict`, *optional*, defaults to `None`)
                Extra dictionary with arguments for precise ensembling control. The following options are available:
                - reduction (`str`, *optional*, defaults to `"closest"`): Defines the ensembling function applied in
                  every pixel location, can be either `"closest"` or `"mean"`.
            latents (`torch.Tensor`, *optional*, defaults to `None`):
                Latent noise tensors to replace the random initialization. These can be taken from the previous
                function call's output.
            generator (`torch.Generator`, or `List[torch.Generator]`, *optional*, defaults to `None`):
                Random number generator object to ensure reproducibility.
            output_type (`str`, *optional*, defaults to `"np"`):
                Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted
                values are: `"np"` (numpy array) or `"pt"` (torch tensor).
            output_uncertainty (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that
                the `ensemble_size` argument is set to a value above 2.
            output_latent (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `latent` field contains the latent codes corresponding to the predictions
                within the ensemble. These codes can be saved, modified, and used for subsequent calls with the
                `latents` argument.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.marigold.MarigoldDepthOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.marigold.MarigoldNormalsOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.marigold.MarigoldNormalsOutput`] is returned, otherwise a
                `tuple` is returned where the first element is the prediction, the second element is the uncertainty
                (or `None`), and the third is the latent (or `None`).
        """

        # 0. Resolving variables.
        device = self._execution_device
        dtype = self.dtype

        # Model-specific optimal default values leading to fast and reasonable results.
        if num_inference_steps is None:
            num_inference_steps = self.default_denoising_steps
        if processing_resolution is None:
            processing_resolution = self.default_processing_resolution

        # 1. Check inputs.
        num_images = self.check_inputs(
            image,
            num_inference_steps,
            ensemble_size,
            processing_resolution,
            resample_method_input,
            resample_method_output,
            batch_size,
            ensembling_kwargs,
            latents,
            generator,
            output_type,
            output_uncertainty,
        )

        self.empty_text_embedding = torch.zeros(1, 257, 1024).to(device, dtype)


        # 4. Preprocess input images. This function loads input image or images of compatible dimensions `(H, W)`,
        # optionally downsamples them to the `processing_resolution` `(PH, PW)`, where
        # `max(PH, PW) == processing_resolution`, and pads the dimensions to `(PPH, PPW)` such that these values are
        # divisible by the latent space downscaling factor (typically 8 in Stable Diffusion). The default value `None`
        # of `processing_resolution` resolves to the optimal value from the model config. It is a recommended mode of
        # operation and leads to the most reasonable results. Using the native image resolution or any other processing
        # resolution can lead to loss of either fine details or global context in the output predictions.
        if not skip_preprocess:
            image, padding, original_resolution = self.image_processor.preprocess(
                image, processing_resolution, resample_method_input, device, dtype
            )  # [N,3,PPH,PPW]
        else:
            padding = (0, 0)
            original_resolution = image.shape[2:]
        # 5. Encode input image into latent space. At this step, each of the `N` input images is represented with `E`
        # ensemble members. Each ensemble member is an independent diffused prediction, just initialized independently.
        # Latents of each such predictions across all input images and all ensemble members are represented in the
        # `pred_latent` variable. The variable `image_latent` is of the same shape: it contains each input image encoded
        # into latent space and replicated `E` times. The latents can be either generated (see `generator` to ensure
        # reproducibility), or passed explicitly via the `latents` argument. The latter can be set outside the pipeline
        # code. For example, in the Marigold-LCM video processing demo, the latents initialization of a frame is taken
        # as a convex combination of the latents output of the pipeline for the previous frame and a newly-sampled
        # noise. This behavior can be achieved by setting the `output_latent` argument to `True`. The latent space
        # dimensions are `(h, w)`. Encoding into latent space happens in batches of size `batch_size`.
        # Model invocation: self.vae.encoder.
        image_latent, pred_latent, gaus_noise = self.prepare_latents(
            image, latents, generator, ensemble_size, batch_size
        )  # [N*E,4,h,w], [N*E,4,h,w]

        del image

        # 6. obtain control_output

        cond_scale =controlnet_conditioning_scale
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            image_latent.detach(),
            self.t_start,
            encoder_hidden_states=self.empty_text_embedding,
            conditioning_scale=cond_scale,
            guess_mode=False,
            return_dict=False,
        )

        # 7. YOSO sampling
        latent_x_t = self.unet(
            pred_latent,
            self.t_start,
            encoder_hidden_states=self.empty_text_embedding,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]


        del (
            pred_latent,
            image_latent,
        )

        # decoder
        prediction = self.decode_prediction(latent_x_t)
        prediction = self.image_processor.unpad_image(prediction, padding)  # [N*E,3,PH,PW]

        prediction = self.image_processor.resize_antialias(
            prediction, original_resolution, resample_method_output, is_aa=False
        )  # [N,3,H,W]
        prediction = self.normalize_normals(prediction)  # [N,3,H,W]

        if output_type == "np":
            prediction = self.image_processor.pt_to_numpy(prediction)  # [N,H,W,3]

        # 11. Offload all models
        self.maybe_free_model_hooks()

        return YosoOutput(
            prediction=prediction,
            latent=latent_x_t,
            gaus_noise=gaus_noise,
        )

    # Copied from diffusers.pipelines.marigold.pipeline_marigold_depth.MarigoldDepthPipeline.prepare_latents
    def prepare_latents(
        self,
        image: torch.Tensor,
        latents: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        ensemble_size: int,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def retrieve_latents(encoder_output):
            if hasattr(encoder_output, "latent_dist"):
                return encoder_output.latent_dist.mode()
            elif hasattr(encoder_output, "latents"):
                return encoder_output.latents
            else:
                raise AttributeError("Could not access latents of provided encoder_output")

        image_latent = torch.cat(
            [
                retrieve_latents(self.vae.encode(image[i : i + batch_size]))
                for i in range(0, image.shape[0], batch_size)
            ],
            dim=0,
        )  # [N,4,h,w]
        image_latent = image_latent * self.vae.config.scaling_factor
        image_latent = image_latent.repeat_interleave(ensemble_size, dim=0)  # [N*E,4,h,w]
        gaus_noise = torch.randn_like(image_latent) 
        pred_latent = image_latent
        return image_latent, pred_latent, gaus_noise

    def decode_prediction(self, pred_latent: torch.Tensor) -> torch.Tensor:
        if pred_latent.dim() != 4 or pred_latent.shape[1] != self.vae.config.latent_channels:
            raise ValueError(
                f"Expecting 4D tensor of shape [B,{self.vae.config.latent_channels},H,W]; got {pred_latent.shape}."
            )

        prediction = self.vae.decode(pred_latent / self.vae.config.scaling_factor, return_dict=False)[0]  # [B,3,H,W]

        prediction = self.normalize_normals(prediction)  # [B,3,H,W]

        return prediction  # [B,3,H,W]

    @staticmethod
    def normalize_normals(normals: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if normals.dim() != 4 or normals.shape[1] != 3:
            raise ValueError(f"Expecting 4D tensor of shape [B,3,H,W]; got {normals.shape}.")

        norm = torch.norm(normals, dim=1, keepdim=True)
        normals /= norm.clamp(min=eps)

        return normals