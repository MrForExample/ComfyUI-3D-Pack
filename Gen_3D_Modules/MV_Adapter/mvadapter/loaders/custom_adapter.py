import os
from typing import Dict, Optional, Union

import safetensors
import torch
from diffusers.utils import _get_model_file, logging
from safetensors import safe_open
from huggingface_hub import snapshot_download

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomAdapterMixin:
    def init_custom_adapter(self, *args, **kwargs):
        self._init_custom_adapter(*args, **kwargs)

    def _init_custom_adapter(self, *args, **kwargs):
        raise NotImplementedError

    def load_custom_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: Optional[str] = None,
        local_cache_dir: Optional[str] = None,
        **kwargs,
    ):
        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # First try to load from local cache if provided
            if local_cache_dir and os.path.exists(os.path.join(local_cache_dir, weight_name)):
                model_file = os.path.join(local_cache_dir, weight_name)
                logger.info(f"Loading adapter from local cache: {model_file}")
            else:
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name,
                        subfolder=subfolder,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        user_agent=user_agent,
                    )
                except (OSError, EnvironmentError) as e:
                    # If file not found and we have a custom cache dir, try to download
                    if local_cache_dir:
                        logger.info(f"Adapter not found. Downloading {pretrained_model_name_or_path_or_dict} to {local_cache_dir}")
                        try:
                            # Create directory if it doesn't exist
                            os.makedirs(local_cache_dir, exist_ok=True)
                            
                            # Download the repository to local cache
                            snapshot_download(
                                repo_id=pretrained_model_name_or_path_or_dict,
                                local_dir=local_cache_dir,
                                ignore_patterns=["*.yaml", "*.json", "*.py", ".png", ".jpg", ".gif"]
                            )
                            
                            # Now try to load from the local cache
                            model_file = os.path.join(local_cache_dir, weight_name)
                            if not os.path.exists(model_file):
                                raise EnvironmentError(f"File {weight_name} not found even after download to {local_cache_dir}")
                            
                        except Exception as download_error:
                            logger.error(f"Failed to download adapter: {download_error}")
                            raise e  # Re-raise original error
                    else:
                        raise e  # Re-raise original error
                
            if weight_name.endswith(".safetensors"):
                state_dict = {}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        self._load_custom_adapter(state_dict)

    def _load_custom_adapter(self, state_dict):
        raise NotImplementedError

    def save_custom_adapter(
        self,
        save_directory: Union[str, os.PathLike],
        weight_name: str,
        safe_serialization: bool = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        if safe_serialization:

            def save_function(weights, filename):
                return safetensors.torch.save_file(
                    weights, filename, metadata={"format": "pt"}
                )

        else:
            save_function = torch.save

        # Save the model
        state_dict = self._save_custom_adapter(**kwargs)
        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(
            f"Custom adapter weights saved in {os.path.join(save_directory, weight_name)}"
        )

    def _save_custom_adapter(self):
        raise NotImplementedError
