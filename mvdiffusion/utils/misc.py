import os
from omegaconf import OmegaConf
from packaging import version
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver('calc_exp_lr_decay_rate', lambda factor, n: factor**(1./n), replace=True)
OmegaConf.register_new_resolver('add', lambda a, b: a + b, replace=True)
OmegaConf.register_new_resolver('sub', lambda a, b: a - b, replace=True)
OmegaConf.register_new_resolver('mul', lambda a, b: a * b, replace=True)
OmegaConf.register_new_resolver('div', lambda a, b: a / b, replace=True)
OmegaConf.register_new_resolver('idiv', lambda a, b: a // b, replace=True)
OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p), replace=True)
# ======================================================= #


def prompt(question):
    inp = input(f"{question} (y/n)").lower().strip()
    if inp and inp == 'y':
        return True
    if inp and inp == 'n':
        return False
    return prompt(question)

@dataclass
class MVConfig:
    pretrained_unet_path: str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int

    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str  # joint, or ablation

    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool

def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    
    schema = OmegaConf.structured(MVConfig)
    conf = OmegaConf.merge(schema, conf)
    return conf

def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def parse_version(ver):
    return version.parse(ver)
