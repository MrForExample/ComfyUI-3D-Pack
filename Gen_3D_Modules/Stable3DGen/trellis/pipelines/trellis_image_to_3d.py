# trellis_image_to_3d.py
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
import gc
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import MeshExtractResult
from contextlib import contextmanager
from typing import Literal
import folder_paths
from stable_trellis_model_manager import TrellisModelManager as StableTrellisModelManager

import os
import logging

logger = logging.getLogger("IF_Hi3DGen")

class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self._device = next(self.models['image_cond_model'].parameters()).device
        self._models_cpu = {}
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str, dinov2_model: str = None) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
            dinov2_model (str, optional): Override the DINOv2 model specified in config
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        # Store model directory path
        new_pipeline.model_dir = os.path.dirname(os.path.dirname(path))  # Go up two levels (past ckpts/)
        
        new_pipeline._models_cpu = {}

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        # Use user-specified model if provided, otherwise use config
        model_name = dinov2_model or args['image_cond_model']
        
        try:
            new_pipeline._init_image_cond_model(model_name)
        except Exception as e:
            logger.error(f"Error initializing image conditioning model: {str(e)}")
            raise

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        if not hasattr(self, 'model_dir'):
            raise AttributeError("Pipeline model_dir not set. Please ensure from_pretrained() is called first.")
            
        # Create model manager instance with proper config
        config = getattr(self, 'config', {})
        model_manager = StableTrellisModelManager(self.model_dir, config=config)
        
        try:
            # This will handle downloading if needed
            dinov2_model = model_manager.load_dinov2(name)
            
            # Ensure model is in consistent dtype based on config
            if getattr(config, 'use_fp16', True):
                dinov2_model = dinov2_model.half()
            else:
                dinov2_model = dinov2_model.float()
                
            self.models['image_cond_model'] = dinov2_model
            
            transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.image_cond_model_transform = transform
            
        except Exception as e:
            logger.error(f"Error loading DINOv2 model: {str(e)}")
            raise

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        # Apply normalization and convert to model dtype
        image = self.image_cond_model_transform(image)
        model_dtype = next(self.models['image_cond_model'].parameters()).dtype
        image = image.to(device=self.device, dtype=model_dtype)
        
        # Run model inference
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    @property
    def device(self) -> torch.device:
        """Override device property to ensure it persists"""
        if not hasattr(self, '_device'):
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device

    def cleanup(self):
        """Explicit cleanup method that preserves essential properties"""
        # Store current device before cleanup
        current_device = self.device
        
        # Clear rembg session
        if hasattr(self, 'rembg_session') and self.rembg_session is not None:
            del self.rembg_session
            self.rembg_session = None
        
        # Clear CPU models
        if hasattr(self, '_models_cpu'):
            for key in list(self._models_cpu.keys()):
                del self._models_cpu[key]
            self._models_cpu.clear()
        
        # Clear GPU models while preserving the models dict
        if hasattr(self, 'models'):
            for key in list(self.models.keys()):
                del self.models[key]
        
        # Force memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Restore essential properties
        self._device = current_device
        if not hasattr(self, 'models'):
            self.models = {}
        if not hasattr(self, '_models_cpu'):
            self._models_cpu = {}

    def unload_models(self, model_keys: List[str]):
        """Unload specific models from GPU memory"""
        for key in model_keys:
            if key in self.models:
                self._models_cpu[key] = self.models[key].cpu()
                del self.models[key]
        torch.cuda.empty_cache()
        gc.collect()

    def load_models(self, model_keys: List[str]):
        """Load specific models back to GPU"""
        for key in model_keys:
            if key in self._models_cpu:
                self.models[key] = self._models_cpu[key].to(self.device)
                # Convert to fp16 if pipeline was configured for it
                if hasattr(self, '_use_fp16') and self._use_fp16 and hasattr(self.models[key], 'half'):
                    self.models[key] = self.models[key].half()
                del self._models_cpu[key]

    @torch.no_grad()
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Load required models
        self.load_models(['sparse_structure_flow_model', 'sparse_structure_decoder'])

        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        
        # Get model dtype for consistency
        model_dtype = next(flow_model.parameters()).dtype
        
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device, dtype=model_dtype)
        
        # Ensure conditioning tensors match model dtype
        cond_converted = {}
        for key, value in cond.items():
            if isinstance(value, torch.Tensor):
                cond_converted[key] = value.to(dtype=model_dtype)
            else:
                cond_converted[key] = value
        
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond_converted,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        for format_type in formats:
            # Load and process one format at a time
            if format_type == 'mesh':
                self.load_models(['slat_decoder_mesh'])
                ret['mesh'] = self.models['slat_decoder_mesh'](slat)
                self.unload_models(['slat_decoder_mesh'])
                
            elif format_type == 'gaussian':
                self.load_models(['slat_decoder_gs'])
                ret['gaussian'] = self.models['slat_decoder_gs'](slat)
                self.unload_models(['slat_decoder_gs'])
                
            elif format_type == 'radiance_field':
                self.load_models(['slat_decoder_rf'])
                ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
                self.unload_models(['slat_decoder_rf'])
            
            # Force garbage collection after each format
            torch.cuda.empty_cache()
            gc.collect()
            
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Load required models
        self.load_models(['slat_flow_model'])
        
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        
        # Get model dtype for consistency
        model_dtype = next(flow_model.parameters()).dtype
        
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device, dtype=model_dtype),
            coords=coords,
        )
        
        # Ensure conditioning tensors match model dtype
        cond_converted = {}
        for key, value in cond.items():
            if isinstance(value, torch.Tensor):
                cond_converted[key] = value.to(dtype=model_dtype)
            else:
                cond_converted[key] = value
        
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond_converted,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device, dtype=slat.feats.dtype)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device, dtype=slat.feats.dtype)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
            
        # Load and process image conditioning
        self.load_models(['image_cond_model'])
        cond = self.get_cond([image])
        self.unload_models(['image_cond_model'])
        
        torch.manual_seed(seed)
        
        # Generate sparse structure
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        torch.cuda.empty_cache()
        gc.collect()
        
        # Generate SLAT
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        torch.cuda.empty_cache()
        gc.collect()

        # Unload all models but don't do full cleanup
        self.unload_models(list(self.models.keys()))
        
        # Process formats one at a time
        results = self.decode_slat(slat, formats)
        torch.cuda.empty_cache()
        gc.collect()
        
        return results

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ):
        """
        Inject a sampler with multiple images as condition.
        
        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f'_old_inference_model', sampler._inference_model)

        if mode == 'stochastic':
            if num_images > num_steps:
                print(f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m")

            cond_indices = (np.arange(num_steps) % num_images).tolist()
            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx:cond_idx+1]
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)
        
        elif mode == 'multidiffusion':
            from .samplers import FlowEulerSampler
            def _new_inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    neg_pred = FlowEulerSampler._inference_model(self, model, x_t, t, neg_cond, **kwargs)
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        preds.append(FlowEulerSampler._inference_model(self, model, x_t, t, cond[i:i+1], **kwargs))
                    pred = sum(preds) / len(preds)
                    return pred
            
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f'_old_inference_model')

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @contextmanager
    def inference_context(self):
        """Context manager for inference that handles device placement and cleanup."""
        try:
            # Move models to device if needed
            if hasattr(self, '_models_cpu'):
                for key in self._models_cpu:
                    if key in self.models:
                        continue
                    self.models[key] = self._models_cpu[key].to(self._device)
                    if hasattr(self.models[key], 'eval'):
                        self.models[key].eval()

            with torch.no_grad():
                yield

        finally:
            # Cleanup
            if hasattr(self, '_models_cpu'):
                for key in list(self.models.keys()):
                    if key in self._models_cpu:
                        self._models_cpu[key] = self.models[key].cpu()
                        del self.models[key]
            
            # Force memory cleanup
            torch.cuda.empty_cache()
            gc.collect()