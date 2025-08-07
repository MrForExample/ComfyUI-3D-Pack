# -*- coding: utf-8 -*-

# Copyright (c) 2012-2015, P. M. Neila
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Utilities for smoothing the occ/sdf grids.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy import sparse

__all__ = [
    "smooth",
    "smooth_constrained",
    "smooth_gaussian",
    "signed_distance_function",
    "smooth_gpu",
    "smooth_constrained_gpu",
    "smooth_gaussian_gpu",
    "signed_distance_function_gpu",
]


def _build_variable_indices(band: np.ndarray) -> np.ndarray:
    num_variables = np.count_nonzero(band)
    variable_indices = np.full(band.shape, -1, dtype=np.int_)
    variable_indices[band] = np.arange(num_variables)
    return variable_indices


def _buildq3d(variable_indices: np.ndarray):
    """
    Builds the filterq matrix for the given variables.
    """

    num_variables = variable_indices.max() + 1
    filterq = sparse.lil_matrix((3 * num_variables, num_variables))

    # Pad variable_indices to simplify out-of-bounds accesses
    variable_indices = np.pad(
        variable_indices, [(0, 1), (0, 1), (0, 1)], mode="constant", constant_values=-1
    )

    coords = np.nonzero(variable_indices >= 0)
    for count, (i, j, k) in enumerate(zip(*coords)):

        assert variable_indices[i, j, k] == count

        filterq[3 * count, count] = -2
        neighbor = variable_indices[i - 1, j, k]
        if neighbor >= 0:
            filterq[3 * count, neighbor] = 1
        else:
            filterq[3 * count, count] += 1

        neighbor = variable_indices[i + 1, j, k]
        if neighbor >= 0:
            filterq[3 * count, neighbor] = 1
        else:
            filterq[3 * count, count] += 1

        filterq[3 * count + 1, count] = -2
        neighbor = variable_indices[i, j - 1, k]
        if neighbor >= 0:
            filterq[3 * count + 1, neighbor] = 1
        else:
            filterq[3 * count + 1, count] += 1

        neighbor = variable_indices[i, j + 1, k]
        if neighbor >= 0:
            filterq[3 * count + 1, neighbor] = 1
        else:
            filterq[3 * count + 1, count] += 1

        filterq[3 * count + 2, count] = -2
        neighbor = variable_indices[i, j, k - 1]
        if neighbor >= 0:
            filterq[3 * count + 2, neighbor] = 1
        else:
            filterq[3 * count + 2, count] += 1

        neighbor = variable_indices[i, j, k + 1]
        if neighbor >= 0:
            filterq[3 * count + 2, neighbor] = 1
        else:
            filterq[3 * count + 2, count] += 1

    filterq = filterq.tocsr()
    return filterq.T.dot(filterq)


def _buildq3d_gpu(variable_indices: torch.Tensor, chunk_size=10000):
    """
    Builds the filterq matrix for the given variables on GPU, using chunking to reduce memory usage.
    """
    device = variable_indices.device
    num_variables = variable_indices.max().item() + 1

    # Pad variable_indices to simplify out-of-bounds accesses
    variable_indices = torch.nn.functional.pad(
        variable_indices, (0, 1, 0, 1, 0, 1), mode="constant", value=-1
    )

    coords = torch.nonzero(variable_indices >= 0)
    i, j, k = coords[:, 0], coords[:, 1], coords[:, 2]

    # Function to process a chunk of data
    def process_chunk(start, end):
        row_indices = []
        col_indices = []
        values = []

        for axis in range(3):
            row_indices.append(3 * torch.arange(start, end, device=device) + axis)
            col_indices.append(
                variable_indices[i[start:end], j[start:end], k[start:end]]
            )
            values.append(torch.full((end - start,), -2, device=device))

            for offset in [-1, 1]:
                if axis == 0:
                    neighbor = variable_indices[
                        i[start:end] + offset, j[start:end], k[start:end]
                    ]
                elif axis == 1:
                    neighbor = variable_indices[
                        i[start:end], j[start:end] + offset, k[start:end]
                    ]
                else:
                    neighbor = variable_indices[
                        i[start:end], j[start:end], k[start:end] + offset
                    ]

                mask = neighbor >= 0
                row_indices.append(
                    3 * torch.arange(start, end, device=device)[mask] + axis
                )
                col_indices.append(neighbor[mask])
                values.append(torch.ones(mask.sum(), device=device))

                # Add 1 to the diagonal for out-of-bounds neighbors
                row_indices.append(
                    3 * torch.arange(start, end, device=device)[~mask] + axis
                )
                col_indices.append(
                    variable_indices[i[start:end], j[start:end], k[start:end]][~mask]
                )
                values.append(torch.ones((~mask).sum(), device=device))

        return torch.cat(row_indices), torch.cat(col_indices), torch.cat(values)

    # Process data in chunks
    all_row_indices = []
    all_col_indices = []
    all_values = []

    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        row_indices, col_indices, values = process_chunk(start, end)
        all_row_indices.append(row_indices)
        all_col_indices.append(col_indices)
        all_values.append(values)

    # Concatenate all chunks
    row_indices = torch.cat(all_row_indices)
    col_indices = torch.cat(all_col_indices)
    values = torch.cat(all_values)

    # Create sparse tensor
    indices = torch.stack([row_indices, col_indices])
    filterq = torch.sparse_coo_tensor(
        indices, values, (3 * num_variables, num_variables)
    )

    # Compute filterq.T @ filterq
    return torch.sparse.mm(filterq.t(), filterq)


# Usage example:
# variable_indices = torch.tensor(...).cuda()  # Your input tensor on GPU
# result = _buildq3d_gpu(variable_indices)


def _buildq2d(variable_indices: np.ndarray):
    """
    Builds the filterq matrix for the given variables.

    Version for 2 dimensions.
    """

    num_variables = variable_indices.max() + 1
    filterq = sparse.lil_matrix((3 * num_variables, num_variables))

    # Pad variable_indices to simplify out-of-bounds accesses
    variable_indices = np.pad(
        variable_indices, [(0, 1), (0, 1)], mode="constant", constant_values=-1
    )

    coords = np.nonzero(variable_indices >= 0)
    for count, (i, j) in enumerate(zip(*coords)):
        assert variable_indices[i, j] == count

        filterq[2 * count, count] = -2
        neighbor = variable_indices[i - 1, j]
        if neighbor >= 0:
            filterq[2 * count, neighbor] = 1
        else:
            filterq[2 * count, count] += 1

        neighbor = variable_indices[i + 1, j]
        if neighbor >= 0:
            filterq[2 * count, neighbor] = 1
        else:
            filterq[2 * count, count] += 1

        filterq[2 * count + 1, count] = -2
        neighbor = variable_indices[i, j - 1]
        if neighbor >= 0:
            filterq[2 * count + 1, neighbor] = 1
        else:
            filterq[2 * count + 1, count] += 1

        neighbor = variable_indices[i, j + 1]
        if neighbor >= 0:
            filterq[2 * count + 1, neighbor] = 1
        else:
            filterq[2 * count + 1, count] += 1

    filterq = filterq.tocsr()
    return filterq.T.dot(filterq)


def _jacobi(
    filterq,
    x0: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    max_iters: int = 10,
    rel_tol: float = 1e-6,
    weight: float = 0.5,
):
    """Jacobi method with constraints."""

    jacobi_r = sparse.lil_matrix(filterq)
    shp = jacobi_r.shape
    jacobi_d = 1.0 / filterq.diagonal()
    jacobi_r.setdiag((0,) * shp[0])
    jacobi_r = jacobi_r.tocsr()

    x = x0

    # We check the stopping criterion each 10 iterations
    check_each = 10
    cum_rel_tol = 1 - (1 - rel_tol) ** check_each

    energy_now = np.dot(x, filterq.dot(x)) / 2
    logging.info("Energy at iter %d: %.6g", 0, energy_now)
    for i in range(max_iters):

        x_1 = -jacobi_d * jacobi_r.dot(x)
        x = weight * x_1 + (1 - weight) * x

        # Constraints.
        x = np.maximum(x, lower_bound)
        x = np.minimum(x, upper_bound)

        # Stopping criterion
        if (i + 1) % check_each == 0:
            # Update energy
            energy_before = energy_now
            energy_now = np.dot(x, filterq.dot(x)) / 2

            logging.info("Energy at iter %d: %.6g", i + 1, energy_now)

            # Check stopping criterion
            cum_rel_improvement = (energy_before - energy_now) / energy_before
            if cum_rel_improvement < cum_rel_tol:
                break

    return x


def signed_distance_function(
    levelset: np.ndarray, band_radius: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the distance to the 0.5 levelset of a function, the mask of the
    border (i.e., the nearest cells to the 0.5 level-set) and the mask of the
    band (i.e., the cells of the function whose distance to the 0.5 level-set
    is less of equal to `band_radius`).
    """

    binary_array = np.where(levelset > 0, True, False)

    # Compute the band and the border.
    dist_func = ndi.distance_transform_edt
    distance = np.where(
        binary_array, dist_func(binary_array) - 0.5, -dist_func(~binary_array) + 0.5
    )
    border = np.abs(distance) < 1
    band = np.abs(distance) <= band_radius

    return distance, border, band


def signed_distance_function_iso0(
    levelset: np.ndarray, band_radius: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the distance to the 0 levelset of a function, the mask of the
    border (i.e., the nearest cells to the 0 level-set) and the mask of the
    band (i.e., the cells of the function whose distance to the 0 level-set
    is less of equal to `band_radius`).
    """

    binary_array = levelset > 0

    # Compute the band and the border.
    dist_func = ndi.distance_transform_edt
    distance = np.where(
        binary_array, dist_func(binary_array), -dist_func(~binary_array)
    )
    border = np.zeros_like(levelset, dtype=bool)
    border[:-1, :, :] |= levelset[:-1, :, :] * levelset[1:, :, :] <= 0
    border[:, :-1, :] |= levelset[:, :-1, :] * levelset[:, 1:, :] <= 0
    border[:, :, :-1] |= levelset[:, :, :-1] * levelset[:, :, 1:] <= 0
    band = np.abs(distance) <= band_radius

    return distance, border, band


def signed_distance_function_gpu(levelset: torch.Tensor, band_radius: int):
    binary_array = (levelset > 0).float()

    # Compute distance transform
    dist_pos = (
        F.max_pool3d(
            -binary_array.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
        )
        .squeeze(0)
        .squeeze(0)
        + binary_array
    )
    dist_neg = F.max_pool3d(
        (binary_array - 1).unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze(0).squeeze(0) + (1 - binary_array)

    distance = torch.where(binary_array > 0, dist_pos - 0.5, -dist_neg + 0.5)

    # breakpoint()

    # Use levelset as distance directly
    # distance = levelset
    # print(distance.shape)
    # Compute border and band
    border = torch.abs(distance) < 1
    band = torch.abs(distance) <= band_radius

    return distance, border, band


def smooth_constrained(
    binary_array: np.ndarray,
    band_radius: int = 4,
    max_iters: int = 250,
    rel_tol: float = 1e-6,
) -> np.ndarray:
    """
    Implementation of the smoothing method from

    "Surface Extraction from Binary Volumes with Higher-Order Smoothness"
    Victor Lempitsky, CVPR10
    """

    # # Compute the distance map, the border and the band.
    logging.info("Computing distance transform...")
    # distance, _, band = signed_distance_function(binary_array, band_radius)
    binary_array_gpu = torch.from_numpy(binary_array).cuda()
    distance, _, band = signed_distance_function_gpu(binary_array_gpu, band_radius)
    distance = distance.cpu().numpy()
    band = band.cpu().numpy()

    variable_indices = _build_variable_indices(band)

    # Compute filterq.
    logging.info("Building matrix filterq...")
    if binary_array.ndim == 3:
        filterq = _buildq3d(variable_indices)
        # variable_indices_gpu = torch.from_numpy(variable_indices).cuda()
        # filterq_gpu = _buildq3d_gpu(variable_indices_gpu)
        # filterq = filterq_gpu.cpu().numpy()
    elif binary_array.ndim == 2:
        filterq = _buildq2d(variable_indices)
    else:
        raise ValueError("binary_array.ndim not in [2, 3]")

    # Initialize the variables.
    res = np.asarray(distance, dtype=np.double)
    x = res[band]
    upper_bound = np.where(x < 0, x, np.inf)
    lower_bound = np.where(x > 0, x, -np.inf)

    upper_bound[np.abs(upper_bound) < 1] = 0
    lower_bound[np.abs(lower_bound) < 1] = 0

    # Solve.
    logging.info("Minimizing energy...")
    x = _jacobi(
        filterq=filterq,
        x0=x,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_iters=max_iters,
        rel_tol=rel_tol,
    )

    res[band] = x
    return res


def total_variation_denoising(x, weight=0.1, num_iterations=5, eps=1e-8):
    diff_x = torch.diff(x, dim=0, prepend=x[:1])
    diff_y = torch.diff(x, dim=1, prepend=x[:, :1])
    diff_z = torch.diff(x, dim=2, prepend=x[:, :, :1])

    norm = torch.sqrt(diff_x**2 + diff_y**2 + diff_z**2 + eps)

    div_x = torch.diff(diff_x / norm, dim=0, append=diff_x[-1:] / norm[-1:])
    div_y = torch.diff(diff_y / norm, dim=1, append=diff_y[:, -1:] / norm[:, -1:])
    div_z = torch.diff(diff_z / norm, dim=2, append=diff_z[:, :, -1:] / norm[:, :, -1:])

    return x - weight * (div_x + div_y + div_z)


def smooth_constrained_gpu(
    binary_array: torch.Tensor,
    band_radius: int = 4,
    max_iters: int = 250,
    rel_tol: float = 1e-4,
):
    distance, _, band = signed_distance_function_gpu(binary_array, band_radius)

    # Initialize variables
    x = distance[band]
    upper_bound = torch.where(x < 0, x, torch.tensor(float("inf"), device=x.device))
    lower_bound = torch.where(x > 0, x, torch.tensor(float("-inf"), device=x.device))

    upper_bound[torch.abs(upper_bound) < 1] = 0
    lower_bound[torch.abs(lower_bound) < 1] = 0

    # Define the 3D Laplacian kernel
    laplacian_kernel = torch.tensor(
        [
            [
                [
                    [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                    [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                ]
            ]
        ],
        device=x.device,
    ).float()

    laplacian_kernel = laplacian_kernel / laplacian_kernel.abs().sum()

    breakpoint()

    # Simplified Jacobi iteration
    for i in range(max_iters):
        # Reshape x to 5D tensor (batch, channel, depth, height, width)
        x_5d = x.view(1, 1, *band.shape)
        x_3d = x.view(*band.shape)

        # Apply 3D convolution
        laplacian = F.conv3d(x_5d, laplacian_kernel, padding=1)

        # Reshape back to original dimensions
        laplacian = laplacian.view(x.shape)

        # Use a small relaxation factor to improve stability
        relaxation_factor = 0.1
        tv_weight = 0.1
        # x_new = x + relaxation_factor * laplacian
        x_new = total_variation_denoising(x_3d, weight=tv_weight)
        # Print laplacian min and max
        # print(f"Laplacian min: {laplacian.min().item():.4f}, max: {laplacian.max().item():.4f}")

        # Apply constraints
        # Reshape x_new to match the dimensions of lower_bound and upper_bound
        x_new = x_new.view(x.shape)
        x_new = torch.clamp(x_new, min=lower_bound, max=upper_bound)

        # Check for convergence
        diff_norm = torch.norm(x_new - x)
        print(diff_norm)
        x_norm = torch.norm(x)

        if x_norm > 1e-8:  # Avoid division by very small numbers
            relative_change = diff_norm / x_norm
            if relative_change < rel_tol:
                break
        elif diff_norm < rel_tol:  # If x_norm is very small, check absolute change
            break

        x = x_new

        # Check for NaN and break if found, also check for inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"NaN or Inf detected at iteration {i}")
            breakpoint()
            break

    result = distance.clone()
    result[band] = x
    return result


def smooth_gaussian(binary_array: np.ndarray, sigma: float = 3) -> np.ndarray:
    vol = np.float_(binary_array) - 0.5
    return ndi.gaussian_filter(vol, sigma=sigma)


def smooth_gaussian_gpu(binary_array: torch.Tensor, sigma: float = 3):
    # vol = binary_array.float()
    vol = binary_array
    kernel_size = int(2 * sigma + 1)
    kernel = torch.ones(
        1,
        1,
        kernel_size,
        kernel_size,
        kernel_size,
        device=binary_array.device,
        dtype=vol.dtype,
    ) / (kernel_size**3)
    return F.conv3d(
        vol.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2
    ).squeeze()


def smooth(binary_array: np.ndarray, method: str = "auto", **kwargs) -> np.ndarray:
    """
    Smooths the 0.5 level-set of a binary array. Returns a floating-point
    array with a smoothed version of the original level-set in the 0 isovalue.

    This function can apply two different methods:

    - A constrained smoothing method which preserves details and fine
      structures, but it is slow and requires a large amount of memory. This
      method is recommended when the input array is small (smaller than
      (500, 500, 500)).
    - A Gaussian filter applied over the binary array. This method is fast, but
      not very precise, as it can destroy fine details. It is only recommended
      when the input array is large and the 0.5 level-set does not contain
      thin structures.

    Parameters
    ----------
    binary_array : ndarray
        Input binary array with the 0.5 level-set to smooth.
    method : str, one of ['auto', 'gaussian', 'constrained']
        Smoothing method. If 'auto' is given, the method will be automatically
        chosen based on the size of `binary_array`.

    Parameters for 'gaussian'
    -------------------------
    sigma : float
        Size of the Gaussian filter (default 3).

    Parameters for 'constrained'
    ----------------------------
    max_iters : positive integer
        Number of iterations of the constrained optimization method
        (default 250).
    rel_tol: float
        Relative tolerance as a stopping criterion (default 1e-6).

    Output
    ------
    res : ndarray
        Floating-point array with a smoothed 0 level-set.
    """

    binary_array = np.asarray(binary_array)

    if method == "auto":
        if binary_array.size > 512**3:
            method = "gaussian"
        else:
            method = "constrained"

    if method == "gaussian":
        return smooth_gaussian(binary_array, **kwargs)

    if method == "constrained":
        return smooth_constrained(binary_array, **kwargs)

    raise ValueError("Unknown method '{}'".format(method))


def smooth_gpu(binary_array: torch.Tensor, method: str = "auto", **kwargs):
    if method == "auto":
        method = "gaussian" if binary_array.numel() > 512**3 else "constrained"

    if method == "gaussian":
        return smooth_gaussian_gpu(binary_array, **kwargs)
    elif method == "constrained":
        return smooth_constrained_gpu(binary_array, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'")
