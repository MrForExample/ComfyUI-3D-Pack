#include <torch/extension.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/Dispatch.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <iostream>

namespace at { namespace native {
namespace {

using namespace at::cuda::detail;

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_2d_grad2_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> grad2_grad_input,
      TensorInfo<scalar_t, index_t> grad2_grad_grid,
      TensorInfo<scalar_t, index_t> grad_output,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_grad_output,
      TensorInfo<scalar_t, index_t> grad_input,
      TensorInfo<scalar_t, index_t> grad_grid,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      const index_t grad_input_memory_span) {

    index_t C = input.sizes[1];
    index_t inp_H = input.sizes[2];
    index_t inp_W = input.sizes[3];

    index_t out_H = grid.sizes[1];
    index_t out_W = grid.sizes[2];

    index_t g2inp_sN = grad2_grad_input.strides[0];
    index_t g2inp_sC = grad2_grad_input.strides[1];
    index_t g2inp_sH = grad2_grad_input.strides[2];
    index_t g2inp_sW = grad2_grad_input.strides[3];

    index_t g2grid_sN = grad2_grad_grid.strides[0];
    index_t g2grid_sH = grad2_grad_grid.strides[1];
    index_t g2grid_sW = grad2_grad_grid.strides[2];
    index_t g2grid_sCoor = grad2_grad_grid.strides[3];

    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sH = grad_output.strides[2];
    index_t gOut_sW = grad_output.strides[3];

    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sH = input.strides[2];
    index_t inp_sW = input.strides[3];

    index_t grid_sN = grid.strides[0];
    index_t grid_sH = grid.strides[1];
    index_t grid_sW = grid.strides[2];
    index_t grid_sCoor = grid.strides[3];

    index_t gInp_sN = grad_input.strides[0];
    index_t gInp_sC = grad_input.strides[1];
    index_t gInp_sH = grad_input.strides[2];
    index_t gInp_sW = grad_input.strides[3];

    index_t gGrid_sW = grad_grid.strides[2];

    index_t ggOut_sN = grad_grad_output.strides[0];
    index_t ggOut_sC = grad_grad_output.strides[1];
    index_t ggOut_sH = grad_grad_output.strides[2];
    index_t ggOut_sW = grad_grad_output.strides[3];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t n = index / (out_H * out_W);

      /* Grid related staff */
      index_t grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t x = grid.data[grid_offset];
      scalar_t y = grid.data[grid_offset + grid_sCoor];

      // multipliers for gradients on ix and iy
      scalar_t gix_mult, giy_mult;
      scalar_t ix = grid_sampler_compute_source_index_set_grad(x, inp_W, padding_mode, align_corners, &gix_mult);
      scalar_t iy = grid_sampler_compute_source_index_set_grad(y, inp_H, padding_mode, align_corners, &giy_mult);

      // get NE, NW, SE, SW pixel values from (x, y)
      index_t ix_nw = static_cast<index_t>(::floor(ix));
      index_t iy_nw = static_cast<index_t>(::floor(iy));
      index_t ix_ne = ix_nw + 1;
      index_t iy_ne = iy_nw;
      index_t ix_sw = ix_nw;
      index_t iy_sw = iy_nw + 1;
      index_t ix_se = ix_nw + 1;
      index_t iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      scalar_t nw = (ix_se - ix)    * (iy_se - iy);
      scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
      scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
      scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);

      /* grad2_grad_input related init */
      scalar_t *g2_inp_ptr_NC = grad2_grad_input.data + n * g2inp_sN;

      /* grad2_grad_grid related init */
      grid_offset = n * g2grid_sN + h * g2grid_sH + w * g2grid_sW;
      scalar_t dx = grad2_grad_grid.data[grid_offset];
      scalar_t dy = grad2_grad_grid.data[grid_offset + g2grid_sCoor];

      dx = dx * gix_mult;
      dy = dy * giy_mult;

      /* grad_output related init */
      scalar_t *gOut_ptr_NCHW = grad_output.data + n * gOut_sN + h * gOut_sH + w * gOut_sW;

      /* input related init */
      scalar_t *inp_ptr_NC = input.data + n * inp_sN;

      /* grad_grad_output related init */
      scalar_t *ggOut_ptr_NCHW = grad_grad_output.data + n * ggOut_sN + h * ggOut_sH + w * ggOut_sW;

      /* grad_input related init */
      index_t NC_offset = n * gInp_sN;

      /* grad_grid related init */
      scalar_t *gGrid_ptr_NHW = grad_grid.data + index * gGrid_sW;
      scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0);

      scalar_t nw_val, ne_val, sw_val, se_val;
      scalar_t g2_nw_val, g2_ne_val, g2_sw_val, g2_se_val;

      scalar_t zero = static_cast<scalar_t>(0);
      for (index_t c = 0; c < C;
           ++c,
           g2_inp_ptr_NC += g2inp_sC,
           inp_ptr_NC += inp_sC,
           NC_offset += gInp_sC,
           gOut_ptr_NCHW += gOut_sC,
           ggOut_ptr_NCHW += ggOut_sC) {

        nw_val = within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)? inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW]: zero;
        ne_val = within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)? inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW]: zero;
        sw_val = within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)? inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW]: zero;
        se_val = within_bounds_2d(iy_se, ix_se, inp_H, inp_W)? inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW]: zero;

        g2_nw_val = within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)? g2_inp_ptr_NC[iy_nw * g2inp_sH + ix_nw * g2inp_sW]: zero;
        g2_ne_val = within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)? g2_inp_ptr_NC[iy_ne * g2inp_sH + ix_ne * g2inp_sW]: zero;
        g2_sw_val = within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)? g2_inp_ptr_NC[iy_sw * g2inp_sH + ix_sw * g2inp_sW]: zero;
        g2_se_val = within_bounds_2d(iy_se, ix_se, inp_H, inp_W)? g2_inp_ptr_NC[iy_se * g2inp_sH + ix_se * g2inp_sW]: zero;

        // Computing gradient wrt to grad_output = grad2_grad_input * x * y + grad2_grad_grid_x * y * val + grad2_grad_grid_y * x * val
        // grad2_grad_input * x * y
        *ggOut_ptr_NCHW = static_cast<scalar_t>(0);
        *ggOut_ptr_NCHW += g2_nw_val * nw + g2_ne_val * ne + g2_sw_val * sw + g2_se_val * se;

        scalar_t nw_tmp = -dx * (iy_se - iy) - dy * (ix_se - ix);
        scalar_t ne_tmp = +dx * (iy_sw - iy) - dy * (ix - ix_sw);
        scalar_t sw_tmp = -dx * (iy - iy_ne) + dy * (ix_ne - ix);
        scalar_t se_tmp = +dx * (iy - iy_nw) + dy * (ix - ix_nw);


        // grad2_grad_grid_x * y * val + grad2_grad_grid_y * x * val
        *ggOut_ptr_NCHW += nw_val * nw_tmp + ne_tmp * ne_val + sw_tmp * sw_val + se_tmp * se_val;

        // Computing gradient wrt input = grad2_grad_grid_x * grad_output * y + grad2_grad_grid_y * grad_output * x
        scalar_t gOut = *gOut_ptr_NCHW;
        //scalar_t val;
        //val = gOut * (-dx  * (iy_se - iy) - dy * (ix_se - ix));
        safe_add_2d(grad_input.data, iy_nw, ix_nw, gInp_sH, gInp_sW, inp_H, inp_W, nw_tmp * gOut, NC_offset, grad_input_memory_span);
        //val = gOut * (+dx * (iy_sw - iy) - dy * (ix - ix_sw));
        safe_add_2d(grad_input.data, iy_ne, ix_ne, gInp_sH, gInp_sW, inp_H, inp_W, ne_tmp * gOut, NC_offset, grad_input_memory_span);
        //val = gOut * (-dx * (iy - iy_ne) + dy * (ix_ne - ix));
        safe_add_2d(grad_input.data, iy_sw, ix_sw, gInp_sH, gInp_sW, inp_H, inp_W, sw_tmp * gOut, NC_offset, grad_input_memory_span);
        //val = gOut * (+dx * (iy - iy_nw) + dy * (ix - ix_nw));
        safe_add_2d(grad_input.data, iy_se, ix_se, gInp_sH, gInp_sW, inp_H, inp_W, se_tmp * gOut, NC_offset, grad_input_memory_span);

        scalar_t dxy = nw_val - ne_val - sw_val + se_val;
        // Computing gradient wrt grid_x = grad2_grad_input * y * gOut + grad2_grad_grid_y * val * gOut
        gix += gOut * (-g2_nw_val * (iy_se - iy) + g2_ne_val * (iy_sw - iy)
                       -g2_sw_val * (iy - iy_ne) + g2_se_val * (iy - iy_nw));
        gix += gOut * dy * dxy;

        // Computing gradient wrt grid_y = grad2_grad_input * x * gOut + grad2_grad_grid_x * val * gOut
        giy += gOut * (-g2_nw_val * (ix_se - ix) - g2_ne_val * (ix - ix_sw)
                       +g2_sw_val * (ix_ne - ix) + g2_se_val * (ix - ix_nw));
        giy += gOut * dx * dxy;
      }

      gGrid_ptr_NHW[0] = gix * gix_mult;
      gGrid_ptr_NHW[1] = giy * giy_mult;
   }
}

template <typename scalar_t, typename index_t>
  C10_LAUNCH_BOUNDS_1(256)
  __global__ void grid_sampler_3d_grad2_kernel(
      const index_t nthreads,
      TensorInfo<scalar_t, index_t> grad2_grad_input,
      TensorInfo<scalar_t, index_t> grad2_grad_grid,
      TensorInfo<scalar_t, index_t> grad_output,
      TensorInfo<scalar_t, index_t> input,
      TensorInfo<scalar_t, index_t> grid,
      TensorInfo<scalar_t, index_t> grad_grad_output,
      TensorInfo<scalar_t, index_t> grad_input,
      TensorInfo<scalar_t, index_t> grad_grid,
      const GridSamplerPadding padding_mode,
      bool align_corners,
      const index_t grad_input_memory_span) {

    index_t C = input.sizes[1];
    index_t inp_D = input.sizes[2];
    index_t inp_H = input.sizes[3];
    index_t inp_W = input.sizes[4];

    index_t out_D = grid.sizes[1];
    index_t out_H = grid.sizes[2];
    index_t out_W = grid.sizes[3];

    index_t g2inp_sN = grad2_grad_input.strides[0];
    index_t g2inp_sC = grad2_grad_input.strides[1];
    index_t g2inp_sD = grad2_grad_input.strides[2];
    index_t g2inp_sH = grad2_grad_input.strides[3];
    index_t g2inp_sW = grad2_grad_input.strides[4];

    index_t g2grid_sN = grad2_grad_grid.strides[0];
    index_t g2grid_sD = grad2_grad_grid.strides[1];
    index_t g2grid_sH = grad2_grad_grid.strides[2];
    index_t g2grid_sW = grad2_grad_grid.strides[3];
    index_t g2grid_sCoor = grad2_grad_grid.strides[4];

    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sD = grad_output.strides[2];
    index_t gOut_sH = grad_output.strides[3];
    index_t gOut_sW = grad_output.strides[4];

    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sD = input.strides[2];
    index_t inp_sH = input.strides[3];
    index_t inp_sW = input.strides[4];

    index_t grid_sN = grid.strides[0];
    index_t grid_sD = grid.strides[1];
    index_t grid_sH = grid.strides[2];
    index_t grid_sW = grid.strides[3];
    index_t grid_sCoor = grid.strides[4];

    index_t gInp_sN = grad_input.strides[0];
    index_t gInp_sC = grad_input.strides[1];
    index_t gInp_sD = grad_input.strides[2];
    index_t gInp_sH = grad_input.strides[3];
    index_t gInp_sW = grad_input.strides[4];

    index_t gGrid_sW = grad_grid.strides[3];

    index_t ggOut_sN = grad_grad_output.strides[0];
    index_t ggOut_sC = grad_grad_output.strides[1];
    index_t ggOut_sD = grad_grad_output.strides[2];
    index_t ggOut_sH = grad_grad_output.strides[3];
    index_t ggOut_sW = grad_grad_output.strides[4];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t d = (index / (out_H * out_W)) % out_D;
      const index_t n = index / (out_D * out_H * out_W);

      /* Grid related staff */
      index_t grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

      // multipliers for gradients on ix and iy
      scalar_t gix_mult, giy_mult, giz_mult;
      ix = grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
      iy = grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
      iz = grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);

      // get NE, NW, SE, SW pixel values from (x, y)
      index_t ix_tnw = static_cast<index_t>(::floor(ix));
      index_t iy_tnw = static_cast<index_t>(::floor(iy));
      index_t iz_tnw = static_cast<index_t>(::floor(iz));

      index_t ix_tne = ix_tnw + 1;
      index_t iy_tne = iy_tnw;
      index_t iz_tne = iz_tnw;

      index_t ix_tsw = ix_tnw;
      index_t iy_tsw = iy_tnw + 1;
      index_t iz_tsw = iz_tnw;

      index_t ix_tse = ix_tnw + 1;
      index_t iy_tse = iy_tnw + 1;
      index_t iz_tse = iz_tnw;

      index_t ix_bnw = ix_tnw;
      index_t iy_bnw = iy_tnw;
      index_t iz_bnw = iz_tnw + 1;

      index_t ix_bne = ix_tnw + 1;
      index_t iy_bne = iy_tnw;
      index_t iz_bne = iz_tnw + 1;

      index_t ix_bsw = ix_tnw;
      index_t iy_bsw = iy_tnw + 1;
      index_t iz_bsw = iz_tnw + 1;

      index_t ix_bse = ix_tnw + 1;
      index_t iy_bse = iy_tnw + 1;
      index_t iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
      scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
      scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
      scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
      scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
      scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
      scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
      scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

      /* grad2_grad_input related init */
      scalar_t *g2_inp_ptr_NC = grad2_grad_input.data + n * g2inp_sN;

      /* grad2_grad_grid related init */
      grid_offset = n * g2grid_sN + d * g2grid_sD + h * g2grid_sH + w * g2grid_sW;
      scalar_t dx = grad2_grad_grid.data[grid_offset];
      scalar_t dy = grad2_grad_grid.data[grid_offset + g2grid_sCoor];
      scalar_t dz = grad2_grad_grid.data[grid_offset + 2 * g2grid_sCoor];

      dx = dx * gix_mult;
      dy = dy * giy_mult;
      dz = dz * giz_mult;

      /* grad_output related init */
      scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;

      /* input related init */
      scalar_t *inp_ptr_NC = input.data + n * inp_sN;

      /* grad_grad_output related init */
      scalar_t *ggOut_ptr_NCDHW = grad_grad_output.data + n * ggOut_sN + d * ggOut_sD + h * ggOut_sH + w * ggOut_sW;

      /* grad_input related init */
      index_t NC_offset = n * gInp_sN;

      /* grad_grid related init */
      scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
      scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);

      scalar_t tnw_val, tne_val, tsw_val, tse_val, bnw_val, bne_val, bsw_val, bse_val;
      scalar_t g2_tnw_val, g2_tne_val, g2_tsw_val, g2_tse_val, g2_bnw_val, g2_bne_val, g2_bsw_val, g2_bse_val;

      scalar_t zero = static_cast<scalar_t>(0);
      for (index_t c = 0; c < C;
           ++c,
           g2_inp_ptr_NC += g2inp_sC,
           inp_ptr_NC += inp_sC,
           NC_offset += gInp_sC,
           gOut_ptr_NCDHW += gOut_sC,
           ggOut_ptr_NCDHW += ggOut_sC) {

        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
          g2_tnw_val = g2_inp_ptr_NC[iz_tnw * g2inp_sD + iy_tnw * g2inp_sH + ix_tnw * g2inp_sW];
        } else {
          tnw_val = zero;
          g2_tnw_val = zero;
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
          g2_tne_val = g2_inp_ptr_NC[iz_tne * g2inp_sD + iy_tne * g2inp_sH + ix_tne * g2inp_sW];
        } else {
          tne_val = zero;
          g2_tne_val = zero;
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
          g2_tsw_val = g2_inp_ptr_NC[iz_tsw * g2inp_sD + iy_tsw * g2inp_sH + ix_tsw * g2inp_sW];
        } else {
          tsw_val = zero;
          g2_tsw_val = zero;
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
          g2_tse_val = g2_inp_ptr_NC[iz_tse * g2inp_sD + iy_tse * g2inp_sH + ix_tse * g2inp_sW];
        } else {
          tse_val = zero;
          g2_tse_val = zero;
        }

        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
          g2_bnw_val = g2_inp_ptr_NC[iz_bnw * g2inp_sD + iy_bnw * g2inp_sH + ix_bnw * g2inp_sW];
        } else {
          bnw_val = zero;
          g2_bnw_val = zero;
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
          g2_bne_val = g2_inp_ptr_NC[iz_bne * g2inp_sD + iy_bne * g2inp_sH + ix_bne * g2inp_sW];
        } else {
          bne_val = zero;
          g2_bne_val = zero;
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
          g2_bsw_val = g2_inp_ptr_NC[iz_bsw * g2inp_sD + iy_bsw * g2inp_sH + ix_bsw * g2inp_sW];
        } else {
          bsw_val = zero;
          g2_bsw_val = zero;
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
          g2_bse_val = g2_inp_ptr_NC[iz_bse * g2inp_sD + iy_bse * g2inp_sH + ix_bse * g2inp_sW];
        } else {
          bse_val = zero;
          g2_bse_val = zero;
        }

        // Computing gradient wrt to grad_output =
        // grad2_grad_input * x * y * z
        *ggOut_ptr_NCDHW = static_cast<scalar_t>(0);
        *ggOut_ptr_NCDHW += g2_tnw_val * tnw + g2_tne_val * tne + g2_tsw_val * tsw + g2_tse_val * tse
                           +g2_bnw_val * bnw + g2_bne_val * bne + g2_bsw_val * bsw + g2_bse_val * bse;

        // +val * (grad2_grad_grid_x * y * z + grad2_grad_grid_y * x * z + grad2_grad_grid_z * x * y)
        scalar_t tnw_tmp = (-dx * (iy_bse - iy) * (iz_bse - iz) - dy * (ix_bse - ix) * (iz_bse - iz) - dz * (ix_bse - ix) * (iy_bse - iy));
        scalar_t tne_tmp = (+dx * (iy_bsw - iy) * (iz_bsw - iz) - dy * (ix - ix_bsw) * (iz_bsw - iz) - dz * (ix - ix_bsw) * (iy_bsw - iy));
        scalar_t tsw_tmp = (-dx * (iy - iy_bne) * (iz_bne - iz) + dy * (ix_bne - ix) * (iz_bne - iz) - dz * (ix_bne - ix) * (iy - iy_bne));
        scalar_t tse_tmp = (+dx * (iy - iy_bnw) * (iz_bnw - iz) + dy * (ix - ix_bnw) * (iz_bnw - iz) - dz * (ix - ix_bnw) * (iy - iy_bnw));
        scalar_t bnw_tmp = (-dx * (iy_tse - iy) * (iz - iz_tse) - dy * (ix_tse - ix) * (iz - iz_tse) + dz * (ix_tse - ix) * (iy_tse - iy));
        scalar_t bne_tmp = (+dx * (iy_tsw - iy) * (iz - iz_tsw) - dy * (ix - ix_tsw) * (iz - iz_tsw) + dz * (ix - ix_tsw) * (iy_tsw - iy));
        scalar_t bsw_tmp = (-dx * (iy - iy_tne) * (iz - iz_tne) + dy * (ix_tne - ix) * (iz - iz_tne) + dz * (ix_tne - ix) * (iy - iy_tne));
        scalar_t bse_tmp = (+dx * (iy - iy_tnw) * (iz - iz_tnw) + dy * (ix - ix_tnw) * (iz - iz_tnw) + dz * (ix - ix_tnw) * (iy - iy_tnw));

        *ggOut_ptr_NCDHW += tnw_val * tnw_tmp + tne_val * tne_tmp + tsw_val * tsw_tmp + tse_val * tse_tmp
                           +bnw_val * bnw_tmp + bne_val * bne_tmp + bsw_val * bsw_tmp + bse_val * bse_tmp;

        // Computing gradient wrt input = grad2_grad_grid_x * grad_output * y * z + grad2_grad_grid_y * grad_output * x * z +
        //                                grad2_grad_grid_z * grad_output * y * z
        scalar_t gOut = *gOut_ptr_NCDHW;

        safe_add_3d(grad_input.data, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw_tmp * gOut,
                    NC_offset, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne_tmp * gOut,
                    NC_offset, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw_tmp * gOut,
                    NC_offset, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse_tmp * gOut,
                    NC_offset, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw_tmp * gOut,
                    NC_offset, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne_tmp * gOut,
                    NC_offset, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw_tmp * gOut,
                    NC_offset, grad_input_memory_span);
        safe_add_3d(grad_input.data, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse_tmp * gOut,
                    NC_offset, grad_input_memory_span);

        //Computing gradient wrt grid
        scalar_t dxy = (tnw_val * (iz_bse - iz) - tne_val * (iz_bsw - iz)
                       -tsw_val * (iz_bne - iz) + tse_val * (iz_bnw - iz)
                       +bnw_val * (iz - iz_tse) - bne_val * (iz - iz_tsw)
                       -bsw_val * (iz - iz_tne) + bse_val * (iz - iz_tnw));

        scalar_t dxz = (tnw_val * (iy_bse - iy) - tne_val * (iy_bsw - iy)
                       +tsw_val * (iy - iy_bne) - tse_val * (iy - iy_bnw)
                       -bnw_val * (iy_tse - iy) + bne_val * (iy_tsw - iy)
                       -bsw_val * (iy - iy_tne) + bse_val * (iy - iy_tnw));

        scalar_t dyz = (tnw_val * (ix_bse - ix) + tne_val * (ix - ix_bsw)
                       -tsw_val * (ix_bne - ix) - tse_val * (ix - ix_bnw)
                       -bnw_val * (ix_tse - ix) - bne_val * (ix - ix_tsw)
                       +bsw_val * (ix_tne - ix) + bse_val * (ix - ix_tnw));


        // Computing gradient wrt grid_x =
        // grad2_grad_input * z * y * gOut
        gix += gOut * (-g2_tnw_val * (iy_bse - iy) * (iz_bse - iz) + g2_tne_val * (iy_bsw - iy) * (iz_bsw - iz)
                       -g2_tsw_val * (iy - iy_bne) * (iz_bne - iz) + g2_tse_val * (iy - iy_bnw) * (iz_bnw - iz)
                       -g2_bnw_val * (iy_tse - iy) * (iz - iz_tse) + g2_bne_val * (iy_tsw - iy) * (iz - iz_tsw)
                       -g2_bsw_val * (iy - iy_tne) * (iz - iz_tne) + g2_bse_val * (iy - iy_tnw) * (iz - iz_tnw));

        //+ grad2_grad_grid_z * y * val * gOut + grad2_grad_grid_y * z * val * gOut
        gix += gOut * (dz * dxz + dy * dxy);

        // Computing gradient wrt grid_y =
        // grad2_grad_input * x * z * gOut
        giy += gOut * (-g2_tnw_val * (ix_bse - ix) * (iz_bse - iz) - g2_tne_val * (ix - ix_bsw) * (iz_bsw - iz)
                       +g2_tsw_val * (ix_bne - ix) * (iz_bne - iz) + g2_tse_val * (ix - ix_bnw) * (iz_bnw - iz)
                       -g2_bnw_val * (ix_tse - ix) * (iz - iz_tse) - g2_bne_val * (ix - ix_tsw) * (iz - iz_tsw)
                       +g2_bsw_val * (ix_tne - ix) * (iz - iz_tne) + g2_bse_val * (ix - ix_tnw) * (iz - iz_tnw));
        //+ grad2_grad_grid_x * z * val * gOut + grad2_grad_grid_z * x * val * gOut
        giy += gOut * (dx * dxy + dz * dyz);

        // Computing gradient wrt grid_z =
        // grad2_grad_input * x * y * gOut
        giz += gOut * (-g2_tnw_val * (ix_bse - ix) * (iy_bse - iy) - g2_tne_val * (ix - ix_bsw) * (iy_bsw - iy)
                       -g2_tsw_val * (ix_bne - ix) * (iy - iy_bne) - g2_tse_val * (ix - ix_bnw) * (iy - iy_bnw)
                       +g2_bnw_val * (ix_tse - ix) * (iy_tse - iy) + g2_bne_val * (ix - ix_tsw) * (iy_tsw - iy)
                       +g2_bsw_val * (ix_tne - ix) * (iy - iy_tne) + g2_bse_val * (ix - ix_tnw) * (iy - iy_tnw));
        //+ grad2_grad_grid_x * y * val * gOut + grad2_grad_grid_y * x * val * gOut
        giz += gOut * (dx * dxz + dy * dyz);
      }

      gGrid_ptr_NDHW[0] = gix * gix_mult;
      gGrid_ptr_NDHW[1] = giy * giy_mult;
      gGrid_ptr_NDHW[2] = giz * giz_mult;
   }
}}


std::vector<torch::Tensor> grid_sample2d_cuda_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners) {

    const auto batch_size = input.size(0);
    const auto C = input.size(1);
    const auto H_IN = input.size(2);
    const auto W_IN = input.size(3);

    const auto H_OUT = grid.size(1);
    const auto W_OUT = grid.size(2);

    torch::Tensor grad_grad_output = torch::zeros_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor grad_grid = torch::zeros_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    int64_t count = batch_size * H_OUT * W_OUT;
 
    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_2d_grad2_cuda", [&] {
          if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
              canUse32BitIndexMath(grad_output)) {
            grid_sampler_2d_grad2_kernel<scalar_t>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<int>(count),
                getTensorInfo<scalar_t, int>(grad2_grad_input),
                getTensorInfo<scalar_t, int>(grad2_grad_grid),
                getTensorInfo<scalar_t, int>(grad_output),
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(grid),
                getTensorInfo<scalar_t, int>(grad_grad_output),
                getTensorInfo<scalar_t, int>(grad_input),
                getTensorInfo<scalar_t, int>(grad_grid),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners,
                static_cast<int>(grad_input.numel()));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            grid_sampler_2d_grad2_kernel<scalar_t>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int64_t>(grad2_grad_input),
                getTensorInfo<scalar_t, int64_t>(grad2_grad_grid),
                getTensorInfo<scalar_t, int64_t>(grad_output),
                getTensorInfo<scalar_t, int64_t>(input),
                getTensorInfo<scalar_t, int64_t>(grid),
                getTensorInfo<scalar_t, int64_t>(grad_grad_output),
                getTensorInfo<scalar_t, int64_t>(grad_input),
                getTensorInfo<scalar_t, int64_t>(grad_grid),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners,
                grad_input.numel());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
    }
  
  return {grad_grad_output, grad_input, grad_grid};
}

std::vector<torch::Tensor> grid_sample3d_cuda_grad2(
    const torch::Tensor &grad2_grad_input,
    const torch::Tensor &grad2_grad_grid,
    const torch::Tensor &grad_output,
    const torch::Tensor &input,
    const torch::Tensor &grid,
    bool padding_mode,
    bool align_corners) {

    const auto batch_size = input.size(0);
    const auto C = input.size(1);
    const auto D_IN = input.size(2);
    const auto H_IN = input.size(3);
    const auto W_IN = input.size(4);

    const auto D_OUT = grid.size(1);
    const auto H_OUT = grid.size(2);
    const auto W_OUT = grid.size(3);

    torch::Tensor grad_grad_output = torch::zeros_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    torch::Tensor grad_grid = torch::zeros_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    int64_t count = batch_size * D_OUT * H_OUT * W_OUT;

    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grid_sampler_3d_grad2_cuda", [&] {
          if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
              canUse32BitIndexMath(grad_output)) {
            grid_sampler_3d_grad2_kernel<scalar_t>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                static_cast<int>(count),
                getTensorInfo<scalar_t, int>(grad2_grad_input),
                getTensorInfo<scalar_t, int>(grad2_grad_grid),
                getTensorInfo<scalar_t, int>(grad_output),
                getTensorInfo<scalar_t, int>(input),
                getTensorInfo<scalar_t, int>(grid),
                getTensorInfo<scalar_t, int>(grad_grad_output),
                getTensorInfo<scalar_t, int>(grad_input),
                getTensorInfo<scalar_t, int>(grad_grid),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners,
                static_cast<int>(grad_input.numel()));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          } else {
            grid_sampler_3d_grad2_kernel<scalar_t>
              <<<GET_BLOCKS(count, 256), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                count,
                getTensorInfo<scalar_t, int64_t>(grad2_grad_input),
                getTensorInfo<scalar_t, int64_t>(grad2_grad_grid),
                getTensorInfo<scalar_t, int64_t>(grad_output),
                getTensorInfo<scalar_t, int64_t>(input),
                getTensorInfo<scalar_t, int64_t>(grid),
                getTensorInfo<scalar_t, int64_t>(grad_grad_output),
                getTensorInfo<scalar_t, int64_t>(grad_input),
                getTensorInfo<scalar_t, int64_t>(grad_grid),
                static_cast<GridSamplerPadding>(padding_mode),
                align_corners,
                grad_input.numel());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          }
        });
    }

  return {grad_grad_output, grad_input, grad_grid};
}

}}
