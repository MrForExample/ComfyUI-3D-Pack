#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

namespace at {namespace native {
std::vector<torch::Tensor> grid_sample2d_cuda_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners);
std::vector<torch::Tensor> grid_sample3d_cuda_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners);
}}

std::vector<torch::Tensor> grid_sample2d_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners) {
  
  return at::native::grid_sample2d_cuda_grad2(grad2_grad_input, grad2_grad_grid,
                                  grad_output, input, grid, padding_mode, align_corners);
}

std::vector<torch::Tensor> grid_sample3d_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners) {

  return at::native::grid_sample3d_cuda_grad2(grad2_grad_input, grad2_grad_grid,
                                  grad_output, input, grid, padding_mode, align_corners);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grad2_2d", &grid_sample2d_grad2, "grid_sample2d second derivative");
  m.def("grad2_3d", &grid_sample3d_grad2, "grid_sample3d second derivative");
}

