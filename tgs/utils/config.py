import os
from dataclasses import dataclass, field

from omegaconf import OmegaConf

from .typing import *

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver("calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n), replace=True)
OmegaConf.register_new_resolver("add", lambda a, b: a + b, replace=True)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b, replace=True)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)
OmegaConf.register_new_resolver("div", lambda a, b: a / b, replace=True)
OmegaConf.register_new_resolver("idiv", lambda a, b: a // b, replace=True)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
OmegaConf.register_new_resolver("rmspace", lambda s, sub: s.replace(" ", sub), replace=True)
OmegaConf.register_new_resolver("tuple2", lambda s: [float(s), float(s)], replace=True)
OmegaConf.register_new_resolver("gt0", lambda s: s > 0, replace=True)
OmegaConf.register_new_resolver("not", lambda s: not s, replace=True)
OmegaConf.register_new_resolver("shsdim", lambda sh_degree: (sh_degree + 1) ** 2 * 3, replace=True)
# ======================================================= #

# ============== Automatic Name Resolvers =============== #
def get_naming_convention(cfg):
    # TODO
    name = f"tgs_{cfg.system.backbone.num_layers}"
    return name

# ======================================================= #

@dataclass
class ExperimentConfig:
    n_gpus: int = 1
    data: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)

def load_config(
    *yamls: str, cli_args: list = [], from_string=False, makedirs=True, **kwargs
) -> Any:
    if from_string:
        parse_func = OmegaConf.create
    else:
        parse_func = OmegaConf.load
    yaml_confs = []
    for y in yamls:
        conf = parse_func(y)
        extends = conf.pop("extends", None)
        if extends:
            assert os.path.exists(extends), f"File {extends} does not exist."
            yaml_confs.append(OmegaConf.load(extends))
        yaml_confs.append(conf)
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg: ExperimentConfig = parse_structured(ExperimentConfig, cfg)

    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg
