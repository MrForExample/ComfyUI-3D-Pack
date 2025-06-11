from typing import Callable, Optional

from .triposg_transformer import TripoSGDiTModel


def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim: Optional[int],
    ori_attn_proc: object,
) -> object:
    return ori_attn_proc


def set_transformer_attn_processor(
    transformer: TripoSGDiTModel,
    set_self_attn_proc_func: Callable = default_set_attn_proc_func,
    set_cross_attn_1_proc_func: Callable = default_set_attn_proc_func,
    set_cross_attn_2_proc_func: Callable = default_set_attn_proc_func,
    set_self_attn_module_names: Optional[list[str]] = None,
    set_cross_attn_1_module_names: Optional[list[str]] = None,
    set_cross_attn_2_module_names: Optional[list[str]] = None,
) -> None:
    do_set_processor = lambda name, module_names: (
        any([name.startswith(module_name) for module_name in module_names])
        if module_names is not None
        else True
    )  # prefix match

    attn_procs = {}
    for name, attn_processor in transformer.attn_processors.items():
        hidden_size = transformer.config.width
        if name.endswith("attn1.processor"):
            # self attention
            attn_procs[name] = (
                set_self_attn_proc_func(name, hidden_size, None, attn_processor)
                if do_set_processor(name, set_self_attn_module_names)
                else attn_processor
            )
        elif name.endswith("attn2.processor"):
            # cross attention
            cross_attention_dim = transformer.config.cross_attention_dim
            attn_procs[name] = (
                set_cross_attn_1_proc_func(
                    name, hidden_size, cross_attention_dim, attn_processor
                )
                if do_set_processor(name, set_cross_attn_1_module_names)
                else attn_processor
            )
        elif name.endswith("attn2_2.processor"):
            # cross attention 2
            cross_attention_dim = transformer.config.cross_attention_2_dim
            attn_procs[name] = (
                set_cross_attn_2_proc_func(
                    name, hidden_size, cross_attention_dim, attn_processor
                )
                if do_set_processor(name, set_cross_attn_2_module_names)
                else attn_processor
            )

    transformer.set_attn_processor(attn_procs)
