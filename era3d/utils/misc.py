import os
from omegaconf import OmegaConf
from packaging import version
from typing import Dict, Optional,  List
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


@dataclass
class TestConfig:
    num_views: int
    dataset: Dict


def load_config(*yaml_files):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    conf = OmegaConf.merge(*yaml_confs)
    OmegaConf.resolve(conf)

    schema = OmegaConf.structured(TestConfig)
    conf = OmegaConf.merge(schema, conf)
    return conf


def config_to_primitive(config, resolve=True):
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path, config):
    with open(path, 'w') as fp:
        OmegaConf.save(config=config, f=fp)

def parse_version(ver):
    return version.parse(ver)
