from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainerSubConfig:
    trainer_type: str = ""
    trainer: dict = field(default_factory=dict)


@dataclass
class ExprimentConfig:
    trainers: List[dict] = field(default_factory=lambda: [])
    init_config: dict = field(default_factory=dict)
    pretrained_model_name_or_path: str = ""
    pretrained_unet_state_dict_path: str = ""
    # expriments related parameters
    linear_beta_schedule: bool = False
    zero_snr: bool = False
    prediction_type: Optional[str] = None
    seed: Optional[int] = None
    max_train_steps: int = 1000000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    mixed_precision: Optional[str] = None       # ["no", "fp16", "bf16", "fp8"]
    skip_training: bool = False
    debug: bool = False