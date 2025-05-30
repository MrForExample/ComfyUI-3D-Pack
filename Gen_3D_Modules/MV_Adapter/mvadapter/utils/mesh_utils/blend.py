import time
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

from .utils import SINGLE_IMAGE_TYPE, image_to_tensor

pb_solver_cpp_source = """
#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void pb_solver_run_cuda(
    torch::Tensor A,
    torch::Tensor X,
    torch::Tensor B,
    torch::Tensor Xbuf,
    int num_iters
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void pb_solver_run(
    torch::Tensor A,
    torch::Tensor X,
    torch::Tensor B,
    torch::Tensor Xbuf,
    int num_iters
) {
    CHECK_INPUT(A);
    CHECK_INPUT(X);
    CHECK_INPUT(B);
    CHECK_INPUT(Xbuf);

    pb_solver_run_cuda(A, X, B, Xbuf, num_iters);
    return;
}

"""

pb_solver_cuda_source = """
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>


__global__ void pb_solver_run_cuda_kernel(
    torch::PackedTensorAccessor32<int64_t,2,torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> X,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> Xbuf
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < A.size(0)) {
        Xbuf[index] = (X[A[index][0]] + X[A[index][1]] + X[A[index][2]] + X[A[index][3]] + B[index]) / 4.;
    }
}

void pb_solver_run_cuda(
    torch::Tensor A,
    torch::Tensor X,
    torch::Tensor B,
    torch::Tensor Xbuf,
    int num_iters
) {
    int batch_size = A.size(0);

    const int threads = 1024;
    const dim3 blocks((batch_size + threads - 1) / threads);
    
    auto A_ptr = A.packed_accessor32<int64_t,2,torch::RestrictPtrTraits>();
    auto X_ptr = X.packed_accessor32<float,1,torch::RestrictPtrTraits>();
    auto B_ptr = B.packed_accessor32<float,1,torch::RestrictPtrTraits>();
    auto Xbuf_ptr = Xbuf.packed_accessor32<float,1,torch::RestrictPtrTraits>();

    for (int i = 0; i < num_iters; ++i) {
        pb_solver_run_cuda_kernel<<<blocks, threads>>>(
            A_ptr,
            X_ptr,
            B_ptr,
            Xbuf_ptr
        );
        cudaDeviceSynchronize();
        std::swap(X_ptr, Xbuf_ptr);
    }
    // we may waste an iteration here, but it's fine
    return;
}
"""


class PBBackend(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def solve(self, num_iters, A, X, B, Xbuf) -> None:
        pass


try:

    @triton.jit
    def pb_triton_step_kernel(
        A_ptr,
        X_ptr,
        B_ptr,
        Xbuf_ptr,
        A_row_stride: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        block_start = pid * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE) + block_start

        A_start_ptr = A_ptr + block_start * A_row_stride
        A_ptrs = A_start_ptr + (
            tl.arange(0, BLOCK_SIZE)[:, None] * A_row_stride
            + tl.arange(0, A_row_stride)[None, :]
        )
        B_ptrs = B_ptr + offsets
        Xbuf_ptrs = Xbuf_ptr + offsets

        mask = offsets < n_elements

        A = tl.load(A_ptrs, mask=mask[:, None], other=0)
        X = tl.load(X_ptr + A)
        B = tl.load(B_ptrs, mask=mask, other=0.0)

        Xout = (tl.sum(X, axis=1) + B) / 4
        tl.store(Xbuf_ptrs, Xout, mask=mask)

except:
    pb_triton_step_kernel = None


class PBTorchCUDAKernelBackend(PBBackend):
    def __init__(self) -> None:
        self.kernel = load_inline(
            name="pb_solver",
            cpp_sources=[pb_solver_cpp_source],
            cuda_sources=[pb_solver_cuda_source],
            functions=["pb_solver_run"],
            verbose=True,
        )

    def solve(self, num_iters, A, X, B, Xbuf) -> None:
        self.kernel.pb_solver_run(A, X, B, Xbuf, num_iters)


class PBTritonBackend(PBBackend):
    def step(
        self, X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Xbuf: torch.Tensor
    ) -> None:
        n_elements = X.shape[0]
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        assert pb_triton_step_kernel is not None
        pb_triton_step_kernel[grid](A, X, B, Xbuf, A.stride(0), n_elements, 1024)

    def solve(self, num_iters, A, X, B, Xbuf) -> None:
        for _ in range(num_iters):
            self.step(X, A, B, Xbuf)
            X, Xbuf = Xbuf, X


class PBTorchNativeBackend(PBBackend):
    def step(
        self, X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Xbuf: torch.Tensor
    ) -> None:
        X.copy_(X[A].sum(-1).add_(B).div_(4))

    def solve(self, num_iters, A, X, B, Xbuf) -> None:
        for _ in range(num_iters):
            self.step(X, A, B, Xbuf)


class PoissonBlendingSolver:
    def __init__(self, backend: str, device: str):
        self.backend = backend
        if backend == "torch-native":
            self.pb_solver = PBTorchNativeBackend()
        elif backend == "torch-cuda":
            self.pb_solver = PBTorchCUDAKernelBackend()
        elif backend == "triton":
            self.pb_solver = PBTritonBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.device = device
        self.lap_kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], device=device, dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.lap_kernel4 = torch.tensor(
            [
                [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            ],
            device=device,
            dtype=torch.float32,
        ).view(4, 1, 3, 3)
        self.neighbor_kernel = torch.tensor(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=device, dtype=torch.float32
        ).view(1, 1, 3, 3)

    def __call__(
        self,
        src: SINGLE_IMAGE_TYPE,
        mask: SINGLE_IMAGE_TYPE,
        tgt: SINGLE_IMAGE_TYPE,
        num_iters: int,
        inplace: bool = True,
        grad_mode: str = "src",
    ):
        src = image_to_tensor(src, device=self.device)
        mask = image_to_tensor(mask, device=self.device)
        tgt = image_to_tensor(tgt, device=self.device)

        assert src.ndim == 3 and tgt.ndim == 3 and mask.ndim in [2, 3]

        if mask.ndim == 3:
            mask = mask.mean(-1) > 0.5
        else:
            mask = mask > 0.5
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0

        tgt_masked = torch.where(mask[..., None], torch.zeros_like(tgt), tgt)

        x, y = mask.nonzero(as_tuple=True)
        N = x.shape[0]
        index_map = torch.cumsum(mask.reshape(-1).long(), dim=-1).reshape(mask.shape)
        index_map[~mask] = 0

        if grad_mode == "src":
            src_lap = F.conv2d(
                src.permute(2, 0, 1)[:, None],
                weight=self.lap_kernel,
                padding=1,
            )[:, 0].permute(1, 2, 0)
            lap = src_lap
        elif grad_mode == "max":
            src_lap = F.conv2d(
                src.permute(2, 0, 1)[:, None],
                weight=self.lap_kernel4,
                padding=1,
            )
            tgt_lap = F.conv2d(
                tgt.permute(2, 0, 1)[:, None],
                weight=self.lap_kernel4,
                padding=1,
            )
            lap = (
                torch.where(src_lap.abs() > tgt_lap.abs(), src_lap, tgt_lap)
                .sum(1)
                .permute(1, 2, 0)
            )
        elif grad_mode == "avg":
            src_lap = F.conv2d(
                src.permute(2, 0, 1)[:, None],
                weight=self.lap_kernel4,
                padding=1,
            )
            tgt_lap = F.conv2d(
                tgt.permute(2, 0, 1)[:, None],
                weight=self.lap_kernel4,
                padding=1,
            )
            lap = (src_lap + tgt_lap).mul(0.5).sum(1).permute(1, 2, 0)

        fq_star = F.conv2d(
            tgt_masked.permute(2, 0, 1)[:, None],
            weight=self.neighbor_kernel,
            padding=1,
        )[:, 0].permute(1, 2, 0)

        A = torch.zeros(N + 1, 4, device=self.device, dtype=torch.long)
        X = torch.zeros(N + 1, 3, device=self.device, dtype=torch.float32)
        B = torch.zeros(N + 1, 3, device=self.device, dtype=torch.float32)

        A[1:] = torch.stack(
            [
                index_map[x - 1, y],
                index_map[x + 1, y],
                index_map[x, y - 1],
                index_map[x, y + 1],
            ],
            dim=-1,
        )
        X[1:] = tgt[x, y]
        B[1:] = lap[x, y] + fq_star[x, y]

        X_flatten = X.flatten()
        B_flatten = B.flatten()
        A_flatten = torch.stack([3 * A, 3 * A + 1, 3 * A + 2], dim=1).reshape(-1, 4)

        buffer = torch.zeros_like(X_flatten)

        self.pb_solver.solve(num_iters, A_flatten, X_flatten, B_flatten, buffer)

        if inplace:
            tgt[x, y] = X_flatten.view(-1, 3)[1:].clamp(0.0, 1.0)
        else:
            tgt = tgt.clone()
            tgt[x, y] = X_flatten.view(-1, 3)[1:].clamp(0.0, 1.0)

        return tgt
