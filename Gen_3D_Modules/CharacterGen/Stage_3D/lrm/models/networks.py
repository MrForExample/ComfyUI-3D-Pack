from dataclasses import dataclass, field
from copy import deepcopy

import torch
import torch.nn as nn
from einops import rearrange

from ..utils.base import BaseModule
from ..utils.ops import get_activation
from ..utils.typing import *


class TriplaneUpsampleNetwork(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 1024
        out_channels: int = 80

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.upsample = nn.ConvTranspose2d(
            self.cfg.in_channels, self.cfg.out_channels, kernel_size=2, stride=2
        )

    def forward(
        self, triplanes: Float[Tensor, "B 3 Ci Hp Wp"]
    ) -> Float[Tensor, "B 3 Co Hp2 Wp2"]:
        triplanes_up = rearrange(
            self.upsample(
                rearrange(triplanes, "B Np Ci Hp Wp -> (B Np) Ci Hp Wp", Np=3)
            ),
            "(B Np) Co Hp Wp -> B Np Co Hp Wp",
            Np=3,
        )
        return triplanes_up


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        bias: bool = True,
        weight_init: Optional[str] = "kaiming_uniform",
        bias_init: Optional[str] = None,
    ):
        super().__init__()
        layers = [
            self.make_linear(
                dim_in,
                n_neurons,
                is_first=True,
                is_last=False,
                bias=bias,
                weight_init=weight_init,
                bias_init=bias_init,
            ),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons,
                    n_neurons,
                    is_first=False,
                    is_last=False,
                    bias=bias,
                    weight_init=weight_init,
                    bias_init=bias_init,
                ),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(
                n_neurons,
                dim_out,
                is_first=False,
                is_last=True,
                bias=bias,
                weight_init=weight_init,
                bias_init=bias_init,
            )
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(output_activation)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(
        self,
        dim_in,
        dim_out,
        is_first,
        is_last,
        bias=True,
        weight_init=None,
        bias_init=None,
    ):
        layer = nn.Linear(dim_in, dim_out, bias=bias)

        if weight_init is None:
            pass
        elif weight_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        else:
            raise NotImplementedError

        if bias:
            if bias_init is None:
                pass
            elif bias_init == "zero":
                torch.nn.init.zeros_(layer.bias)
            else:
                raise NotImplementedError

        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError


@dataclass
class HeadSpec:
    name: str
    out_channels: int
    n_hidden_layers: int
    output_activation: Optional[str] = None


class MultiHeadMLP(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 0
        n_neurons: int = 0
        n_hidden_layers_share: int = 0
        heads: List[HeadSpec] = field(default_factory=lambda: [])
        activation: str = "relu"
        bias: bool = True
        weight_init: Optional[str] = "kaiming_uniform"
        bias_init: Optional[str] = None
        chunk_mode: Optional[str] = None
        chunk_size: int = -1

    cfg: Config

    def configure(self) -> None:
        super().configure()
        shared_layers = [
            self.make_linear(
                self.cfg.in_channels,
                self.cfg.n_neurons,
                bias=self.cfg.bias,
                weight_init=self.cfg.weight_init,
                bias_init=self.cfg.bias_init,
            ),
            self.make_activation(self.cfg.activation),
        ]
        for i in range(self.cfg.n_hidden_layers_share - 1):
            shared_layers += [
                self.make_linear(
                    self.cfg.n_neurons,
                    self.cfg.n_neurons,
                    bias=self.cfg.bias,
                    weight_init=self.cfg.weight_init,
                    bias_init=self.cfg.bias_init,
                ),
                self.make_activation(self.cfg.activation),
            ]
        self.shared_layers = nn.Sequential(*shared_layers)

        assert len(self.cfg.heads) > 0
        heads = {}
        for head in self.cfg.heads:
            head_layers = []
            for i in range(head.n_hidden_layers):
                head_layers += [
                    self.make_linear(
                        self.cfg.n_neurons,
                        self.cfg.n_neurons,
                        bias=self.cfg.bias,
                        weight_init=self.cfg.weight_init,
                        bias_init=self.cfg.bias_init,
                    ),
                    self.make_activation(self.cfg.activation),
                ]
            head_layers += [
                self.make_linear(
                    self.cfg.n_neurons,
                    head.out_channels,
                    bias=self.cfg.bias,
                    weight_init=self.cfg.weight_init,
                    bias_init=self.cfg.bias_init,
                ),
            ]
            heads[head.name] = nn.Sequential(*head_layers)
        self.heads = nn.ModuleDict(heads)

        if self.cfg.chunk_mode is not None:
            assert self.cfg.chunk_size > 0

    def make_linear(
        self,
        dim_in,
        dim_out,
        bias=True,
        weight_init=None,
        bias_init=None,
    ):
        layer = nn.Linear(dim_in, dim_out, bias=bias)

        if weight_init is None:
            pass
        elif weight_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        else:
            raise NotImplementedError

        if bias:
            if bias_init is None:
                pass
            elif bias_init == "zero":
                torch.nn.init.zeros_(layer.bias)
            else:
                raise NotImplementedError

        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError

    def forward(
        self, x, include: Optional[List] = None, exclude: Optional[List] = None
    ):
        inp_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        if self.cfg.chunk_mode is None:
            shared_features = self.shared_layers(x)
        elif self.cfg.chunk_mode == "deferred":
            shared_features = DeferredFunc.apply(
                self.shared_layers, x, self.cfg.chunk_size
            )
        elif self.cfg.chunk_mode == "checkpointing":
            shared_features = apply_batch_checkpointing(
                self.shared_layers, x, self.cfg.chunk_size
            )
        else:
            raise NotImplementedError

        shared_features = shared_features.reshape(*inp_shape, -1)

        if include is not None and exclude is not None:
            raise ValueError("Cannot specify both include and exclude.")
        if include is not None:
            heads = [h for h in self.cfg.heads if h.name in include]
        elif exclude is not None:
            heads = [h for h in self.cfg.heads if h.name not in exclude]
        else:
            heads = self.cfg.heads

        out = {
            head.name: get_activation(head.output_activation)(
                self.heads[head.name](shared_features)
            )
            for head in heads
        }
        """
        # TypeError
        if self.cfg.chunk_mode is None:
            out = {
                head.name: get_activation(head.output_activation)(
                    self.heads[head.name](shared_features)
                )
                for head in heads
            }
        elif self.cfg.chunk_mode  == "deferred":
            out = {
                head.name: get_activation(head.output_activation)(
                    DeferredFunc.apply(self.heads[head.name], shared_features, self.cfg.chunk_size)
                )
                for head in heads
            }
        else:
            raise NotImplementedError
        """
        return out


class DeferredFunc(torch.autograd.Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, model, x, chunk_size):
        model_copy = deepcopy(model)
        model_copy.requires_grad_(False)

        ret = []
        x_split = torch.split(x, chunk_size, dim=0)

        with torch.no_grad():
            for cur_x in x_split:
                ret.append(model_copy(cur_x))

        ctx.model = model
        ctx.save_for_backward(x.detach(), torch.as_tensor(chunk_size))

        ret = torch.cat(ret, dim=0)

        return ret

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        model = ctx.model
        x, chunk_size = ctx.saved_tensors
        chunk_size = chunk_size.item()

        model_copy = deepcopy(model)

        x_split = torch.split(x, chunk_size, dim=0)
        grad_output_split = torch.split(grad_output, chunk_size, 0)
        grad_input_split = []

        with torch.set_grad_enabled(True):
            model_copy.requires_grad_(True)
            model_copy.zero_grad()
            for cur_x, cur_grad_output in zip(x_split, grad_output_split):
                cur_x.requires_grad_(True)
                cur_y = model_copy(cur_x)
                cur_y.backward(cur_grad_output)

                grad_input_split.append(cur_x.grad.clone())

        grad_input = torch.cat(grad_input_split, dim=0)

        model_copy_params = list(model_copy.parameters())
        model_params = list(model.parameters())

        for param, param_copy in zip(model_params, model_copy_params):
            if param.grad is None:
                param.grad = param_copy.grad.clone()
            else:
                param.grad.add_(param_copy.grad)

        return None, grad_input, None


def apply_batch_checkpointing(func, x, chunk_size):
    if chunk_size >= len(x):
        # return func(x)
        return torch.utils.checkpoint.checkpoint(func, x, use_reentrant=False)

    x_split = torch.split(x, chunk_size, dim=0)

    def cat_and_query(y_all, x):
        return torch.cat([y_all, func(x)])

    y_all = func(x_split[0])
    for cur_x in x_split[1:]:
        y_all = torch.utils.checkpoint.checkpoint(
            cat_and_query, y_all, cur_x, use_reentrant=False
        )

    return y_all


def get_encoding(n_input_dims: int, config) -> nn.Module:
    raise NotImplementedError


def get_mlp(n_input_dims, n_output_dims, config) -> nn.Module:
    raise NotImplementedError
