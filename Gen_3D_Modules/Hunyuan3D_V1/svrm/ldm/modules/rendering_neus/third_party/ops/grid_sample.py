from torch.utils.cpp_extension import load
import torch
from pkg_resources import parse_version

from .. import custom_ops
import os

# gridsample_grad2 = load(name='gridsample_grad2', sources=['third_party/ops/gridsample_cuda.cpp', 'third_party/ops/gridsample_cuda.cu'], verbose=True)

gridsample_grad2 = load(name='gridsample_grad2', sources=[os.path.join(os.path.dirname(__file__), f) for f in ['grid_sample.cpp', 'grid_sample.cu']], verbose=True)

gridsample_grad2 = None

def _init():
    global gridsample_grad2
    if gridsample_grad2 is None:
        gridsample_grad2 = custom_ops.get_plugin(
            module_name='gridsample_grad2',
            sources=['gridsample_cuda.cpp', 'gridsample_cuda.cu'],
            headers=None,
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math'],
        )
    return True

_init()

def grid_sample_2d(input, grid, padding_mode='zeros', align_corners=True):
    assert padding_mode in ['zeros', 'border']
    return _GridSample2dForward.apply(input, grid, padding_mode, align_corners)


def grid_sample_3d(input, grid, padding_mode='zeros', align_corners=True):
    assert padding_mode in ['zeros', 'border']
    return _GridSample3dForward.apply(input, grid, padding_mode, align_corners)


_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version('1.11.0a')


class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode=0, align_corners=True):
        assert input.ndim == 4
        assert grid.ndim == 4
        assert input.shape[0] == grid.shape[0]
        assert grid.shape[3] == 2

        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear',
                                                 padding_mode=padding_mode, align_corners=align_corners)
        ctx.save_for_backward(input, grid)
        ctx.padding_mode = ['zeros', 'border'].index(padding_mode)
        ctx.align_corners = align_corners
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(grad_output, input, grid, ctx.padding_mode, ctx.align_corners)
        return grad_input, grad_grid, None, None

class _GridSample2dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid, padding_mode=0, align_corners=True):
        op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')[0]
        if _use_pytorch_1_11_api:
            output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])
            # breakpoint()
            grad_input, grad_grid = op(grad_output, input, grid, 0, padding_mode, align_corners, output_mask)
        else:
            grad_input, grad_grid = op(grad_output, input, grid, 0, padding_mode, align_corners)
        
        ctx.save_for_backward(grad_output, input, grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
 
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        grad_output, input, grid = ctx.saved_tensors
        assert grad_output.is_cuda and input.is_cuda and grid.is_cuda and grad2_grad_input.is_cuda and grad2_grad_grid.is_cuda
        out = gridsample_grad2.grad2_2d(grad2_grad_input, grad2_grad_grid, grad_output,
                                        input, grid, ctx.padding_mode, ctx.align_corners)

        grad_grad_output = out[0]
        grad_input = out[1]
        grad_grid = out[2]

        return grad_grad_output, grad_input, grad_grid, None, None


class _GridSample3dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode=0, align_corners=True):
        assert input.ndim == 5
        assert grid.ndim == 5
        assert input.shape[0] == grid.shape[0]
        assert grid.shape[4] == 3

        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear',
                                                 padding_mode=padding_mode, align_corners=align_corners)
        ctx.save_for_backward(input, grid)
        ctx.padding_mode = ['zeros', 'border'].index(padding_mode)
        ctx.align_corners = align_corners
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample3dBackward.apply(grad_output, input, grid, ctx.padding_mode, ctx.align_corners)
        return grad_input, grad_grid, None, None



class _GridSample3dBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid, padding_mode=0, align_corners=True):
        op = torch._C._jit_get_operation('aten::grid_sampler_3d_backward')
        if _use_pytorch_1_11_api:
            output_mask = (ctx.needs_input_grad[1], ctx.needs_input_grad[2])
            grad_input, grad_grid = op(grad_output, input, grid, 0, padding_mode, align_corners, output_mask)
        else:
            grad_input, grad_grid = op(grad_output, input, grid, 0, padding_mode, align_corners)
        
        ctx.save_for_backward(grad_output, input, grid)
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
 
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        grad_output, input, grid = ctx.saved_tensors
        assert grad_output.is_cuda and input.is_cuda and grid.is_cuda and grad2_grad_input.is_cuda and grad2_grad_grid.is_cuda
        out = gridsample_grad2.grad2_3d(grad2_grad_input, grad2_grad_grid, grad_output,
                                        input, grid, ctx.padding_mode, ctx.align_corners)

        grad_grad_output = out[0]
        grad_input = out[1]
        grad_grid = out[2]

        return grad_grad_output, grad_input, grad_grid, None, None


