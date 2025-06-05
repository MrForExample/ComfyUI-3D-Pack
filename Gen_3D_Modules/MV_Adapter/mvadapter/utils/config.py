import os
from dataclasses import dataclass, field
from datetime import datetime

from omegaconf import OmegaConf

from .core import debug, find, info, warn
from .typing import *

# ============ Register OmegaConf Resolvers ============= #
OmegaConf.register_new_resolver(
    "calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n)
)
OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("idiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
OmegaConf.register_new_resolver("rmspace", lambda s, sub: s.replace(" ", sub))
OmegaConf.register_new_resolver("tuple2", lambda s: [float(s), float(s)])
OmegaConf.register_new_resolver("gt0", lambda s: s > 0)
OmegaConf.register_new_resolver("not", lambda s: not s)


def calc_num_train_steps(num_data, batch_size, max_epochs, num_nodes, num_cards=8):
    return int(num_data / (num_nodes * num_cards * batch_size)) * max_epochs


OmegaConf.register_new_resolver("calc_num_train_steps", calc_num_train_steps)

# ======================================================= #


# ============== Automatic Name Resolvers =============== #
def get_naming_convention(cfg):
    # TODO
    name = f"lrm_{cfg.system.backbone.num_layers}"
    return name


# ======================================================= #


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    tag: str = ""
    seed: int = 0
    use_timestamp: bool = True
    timestamp: Optional[str] = None
    exp_root_dir: str = "outputs"

    ### these shouldn't be set manually
    exp_dir: str = "outputs/default"
    trial_name: str = "exp"
    trial_dir: str = "outputs/default/exp"
    n_gpus: int = 1
    ###

    resume: Optional[str] = None

    data_cls: str = ""
    data: dict = field(default_factory=dict)

    system_cls: str = ""
    system: dict = field(default_factory=dict)

    # accept pytorch-lightning trainer parameters
    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
    trainer: dict = field(default_factory=dict)

    # accept pytorch-lightning checkpoint callback parameters
    # see https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint
    checkpoint: dict = field(default_factory=dict)


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

    # post processing
    # auto naming
    if scfg.name == "auto":
        scfg.name = get_naming_convention(scfg)
    # add timestamp
    if not scfg.tag and not scfg.use_timestamp:
        raise ValueError("Either tag is specified or use_timestamp is True.")
    scfg.trial_name = scfg.tag
    # if resume from an existing config, scfg.timestamp should not be None
    if scfg.timestamp is None:
        scfg.timestamp = ""
        if scfg.use_timestamp:
            if scfg.n_gpus > 1:
                warn(
                    "Timestamp is disabled when using multiple GPUs, please make sure you have a unique tag."
                )
            else:
                scfg.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
    # make directories
    scfg.trial_name += scfg.timestamp
    scfg.exp_dir = os.path.join(scfg.exp_root_dir, scfg.name)
    scfg.trial_dir = os.path.join(scfg.exp_dir, scfg.trial_name)

    if makedirs:
        os.makedirs(scfg.trial_dir, exist_ok=True)

    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.merge(OmegaConf.structured(fields), cfg)
    return scfg
