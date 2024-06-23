from omegaconf import DictConfig, OmegaConf


def parse_structured(fields, cfg) -> DictConfig:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg


def load_config(fields, config, extras=None):
    if extras is not None:
        print("Warning! extra parameter in cli is not verified, may cause erros.")
    if isinstance(config, str):
        cfg = OmegaConf.load(config)
    elif isinstance(config, dict):
        cfg = OmegaConf.create(config)
    elif isinstance(config, DictConfig):
        cfg = config
    else:
        raise NotImplementedError(f"Unsupported config type {type(config)}")
    if extras is not None:
        cli_conf = OmegaConf.from_cli(extras)
        cfg = OmegaConf.merge(cfg, cli_conf)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return parse_structured(fields, cfg)