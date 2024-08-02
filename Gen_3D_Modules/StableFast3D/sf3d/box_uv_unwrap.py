import math
from typing import Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer
from torch import Tensor

from StableFast3D.sf3d.models.utils import dot, triangle_intersection_2d


def _box_assign_vertex_to_cube_face(
    vertex_positions: Float[Tensor, "Nv 3"],
    vertex_normals: Float[Tensor, "Nv 3"],
    triangle_idxs: Integer[Tensor, "Nf 3"],
    bbox: Float[Tensor, "2 3"],
) -> Tuple[Float[Tensor, "Nf 3 2"], Integer[Tensor, "Nf 3"]]:
    # Test to not have a scaled model to fit the space better
    # bbox_min = bbox[:1].mean(-1, keepdim=True)
    # bbox_max = bbox[1:].mean(-1, keepdim=True)
    # v_pos_normalized = (vertex_positions - bbox_min) / (bbox_max - bbox_min)

    # Create a [0, 1] normalized vertex position
    v_pos_normalized = (vertex_positions - bbox[:1]) / (bbox[1:] - bbox[:1])
    # And to [-1, 1]
    v_pos_normalized = 2.0 * v_pos_normalized - 1.0

    # Get all vertex positions for each triangle
    # Now how do we define to which face the triangle belongs? Mean face pos? Max vertex pos?
    v0 = v_pos_normalized[triangle_idxs[:, 0]]
    v1 = v_pos_normalized[triangle_idxs[:, 1]]
    v2 = v_pos_normalized[triangle_idxs[:, 2]]
    tri_stack = torch.stack([v0, v1, v2], dim=1)

    vn0 = vertex_normals[triangle_idxs[:, 0]]
    vn1 = vertex_normals[triangle_idxs[:, 1]]
    vn2 = vertex_normals[triangle_idxs[:, 2]]
    tri_stack_nrm = torch.stack([vn0, vn1, vn2], dim=1)

    # Just average the normals per face
    face_normal = F.normalize(torch.sum(tri_stack_nrm, 1), eps=1e-6, dim=-1)

    # Now decide based on the face normal in which box map we project
    # abs_x, abs_y, abs_z = tri_stack_nrm.abs().unbind(-1)
    abs_x, abs_y, abs_z = tri_stack.abs().unbind(-1)

    axis = torch.tensor(
        [
            [1, 0, 0],  # 0
            [-1, 0, 0],  # 1
            [0, 1, 0],  # 2
            [0, -1, 0],  # 3
            [0, 0, 1],  # 4
            [0, 0, -1],  # 5
        ],
        device=face_normal.device,
        dtype=face_normal.dtype,
    )
    face_normal_axis = (face_normal[:, None] * axis[None]).sum(-1)
    index = face_normal_axis.argmax(-1)

    max_axis, uc, vc = (
        torch.ones_like(abs_x),
        torch.zeros_like(tri_stack[..., :1]),
        torch.zeros_like(tri_stack[..., :1]),
    )
    mask_pos_x = index == 0
    max_axis[mask_pos_x] = abs_x[mask_pos_x]
    uc[mask_pos_x] = tri_stack[mask_pos_x][..., 1:2]
    vc[mask_pos_x] = -tri_stack[mask_pos_x][..., -1:]

    mask_neg_x = index == 1
    max_axis[mask_neg_x] = abs_x[mask_neg_x]
    uc[mask_neg_x] = tri_stack[mask_neg_x][..., 1:2]
    vc[mask_neg_x] = -tri_stack[mask_neg_x][..., -1:]

    mask_pos_y = index == 2
    max_axis[mask_pos_y] = abs_y[mask_pos_y]
    uc[mask_pos_y] = tri_stack[mask_pos_y][..., 0:1]
    vc[mask_pos_y] = -tri_stack[mask_pos_y][..., -1:]

    mask_neg_y = index == 3
    max_axis[mask_neg_y] = abs_y[mask_neg_y]
    uc[mask_neg_y] = tri_stack[mask_neg_y][..., 0:1]
    vc[mask_neg_y] = -tri_stack[mask_neg_y][..., -1:]

    mask_pos_z = index == 4
    max_axis[mask_pos_z] = abs_z[mask_pos_z]
    uc[mask_pos_z] = tri_stack[mask_pos_z][..., 0:1]
    vc[mask_pos_z] = tri_stack[mask_pos_z][..., 1:2]

    mask_neg_z = index == 5
    max_axis[mask_neg_z] = abs_z[mask_neg_z]
    uc[mask_neg_z] = tri_stack[mask_neg_z][..., 0:1]
    vc[mask_neg_z] = -tri_stack[mask_neg_z][..., 1:2]

    # UC from [-1, 1] to [0, 1]
    max_dim_div = max_axis.max(dim=0, keepdims=True).values
    uc = ((uc[..., 0] / max_dim_div + 1.0) * 0.5).clip(0, 1)
    vc = ((vc[..., 0] / max_dim_div + 1.0) * 0.5).clip(0, 1)

    uv = torch.stack([uc, vc], dim=-1)

    return uv, index


def _assign_faces_uv_to_atlas_index(
    vertex_positions: Float[Tensor, "Nv 3"],
    triangle_idxs: Integer[Tensor, "Nf 3"],
    face_uv: Float[Tensor, "Nf 3 2"],
    face_index: Integer[Tensor, "Nf 3"],
) -> Integer[Tensor, "Nf"]:  # noqa: F821
    triangle_pos = vertex_positions[triangle_idxs]
    # We need to do perform 3 overlap checks.
    # The first set is placed in the upper two thirds of the UV atlas.
    # Conceptually, this is the direct visible surfaces from the each cube side
    # The second set is placed in the lower thirds and the left half of the UV atlas.
    # This is the first set of occluded surfaces. They will also be saved in the projected fashion
    # The third pass finds all non assigned faces. They will be placed in the bottom right half of
    # the UV atlas in scattered fashion.
    assign_idx = face_index.clone()
    for overlap_step in range(3):
        overlapping_indicator = torch.zeros_like(assign_idx, dtype=torch.bool)
        for i in range(overlap_step * 6, (overlap_step + 1) * 6):
            mask = assign_idx == i
            if not mask.any():
                continue
            # Get all elements belonging to the projection face
            uv_triangle = face_uv[mask]
            cur_triangle_pos = triangle_pos[mask]
            # Find the center of the uv coordinates
            center_uv = uv_triangle.mean(dim=1, keepdim=True)
            # And also the radius of the triangle
            uv_triangle_radius = (uv_triangle - center_uv).norm(dim=-1).max(-1).values

            potentially_overlapping_mask = (
                # Find all close triangles
                (center_uv[None, ...] - center_uv[:, None]).norm(dim=-1)
                # Do not select the same element by offseting with an large valued identity matrix
                + torch.eye(
                    uv_triangle.shape[0],
                    device=uv_triangle.device,
                    dtype=uv_triangle.dtype,
                ).unsqueeze(-1)
                * 1000
            )
            # Mark all potentially overlapping triangles to reduce the number of triangle intersection tests
            potentially_overlapping_mask = (
                potentially_overlapping_mask
                <= (uv_triangle_radius.view(-1, 1, 1) * 3.0)
            ).squeeze(-1)
            overlap_coords = torch.stack(torch.where(potentially_overlapping_mask), -1)

            # Only unique triangles (A|B and B|A should be the same)
            f = torch.min(overlap_coords, dim=-1).values
            s = torch.max(overlap_coords, dim=-1).values
            overlap_coords = torch.unique(torch.stack([f, s], dim=1), dim=0)
            first, second = overlap_coords.unbind(-1)

            # Get the triangles
            tri_1 = uv_triangle[first]
            tri_2 = uv_triangle[second]

            # Perform the actual set with the reduced number of potentially overlapping triangles
            its = triangle_intersection_2d(tri_1, tri_2, eps=1e-6)

            # So we now need to detect which triangles are the occluded ones.
            # We always assume the first to be the visible one (the others should move)
            # In the previous step we use a lexigraphical sort to get the unique pairs
            # In this we use a sort based on the orthographic projection
            ax = 0 if i < 2 else 1 if i < 4 else 2
            use_max = i % 2 == 1

            tri1_c = cur_triangle_pos[first].mean(dim=1)
            tri2_c = cur_triangle_pos[second].mean(dim=1)

            mark_first = (
                (tri1_c[..., ax] > tri2_c[..., ax])
                if use_max
                else (tri1_c[..., ax] < tri2_c[..., ax])
            )
            first[mark_first] = second[mark_first]

            # Lastly the same index can be tested multiple times.
            # If one marks it as overlapping we keep it marked as such.
            # We do this by testing if it has been marked at least once.
            unique_idx, rev_idx = torch.unique(first, return_inverse=True)

            add = torch.zeros_like(unique_idx, dtype=torch.float32)
            add.index_add_(0, rev_idx, its.float())
            its_mask = add > 0

            # And fill it in the overlapping indicator
            idx = torch.where(mask)[0][unique_idx]
            overlapping_indicator[idx] = its_mask

        # Move the index to the overlap regions (shift by 6)
        assign_idx[overlapping_indicator] += 6

    # We do not care about the correct face placement after the first 2 slices
    max_idx = 6 * 2
    return assign_idx.clamp(0, max_idx)


def _find_slice_offset_and_scale(
    index: Integer[Tensor, "Nf"],  # noqa: F821
) -> Tuple[
    Float[Tensor, "Nf"], Float[Tensor, "Nf"], Float[Tensor, "Nf"], Float[Tensor, "Nf"]  # noqa: F821
]:  # noqa: F821
    # 6 due to the 6 cube faces
    off = 1 / 3
    dupl_off = 1 / 6

    # Here, we need to decide how to pack the textures in the case of overlap
    def x_offset_calc(x, i):
        offset_calc = i // 6
        # Initial coordinates - just 3x2 grid
        if offset_calc == 0:
            return off * x
        else:
            # Smaller 3x2 grid plus eventual shift to right for
            # second overlap
            return dupl_off * x + min(offset_calc - 1, 1) * 0.5

    def y_offset_calc(x, i):
        offset_calc = i // 6
        # Initial coordinates - just a 3x2 grid
        if offset_calc == 0:
            return off * x
        else:
            # Smaller coordinates in the lowest row
            return dupl_off * x + off * 2

    offset_x = torch.zeros_like(index, dtype=torch.float32)
    offset_y = torch.zeros_like(index, dtype=torch.float32)
    offset_x_vals = [0, 1, 2, 0, 1, 2]
    offset_y_vals = [0, 0, 0, 1, 1, 1]
    for i in range(index.max().item() + 1):
        mask = index == i
        if not mask.any():
            continue
        offset_x[mask] = x_offset_calc(offset_x_vals[i % 6], i)
        offset_y[mask] = y_offset_calc(offset_y_vals[i % 6], i)

    div_x = torch.full_like(index, 6 // 2, dtype=torch.float32)
    # All overlap elements are saved in half scale
    div_x[index >= 6] = 6
    div_y = div_x.clone()  # Same for y
    # Except for the random overlaps
    div_x[index >= 12] = 2
    # But the random overlaps are saved in a large block in the lower thirds
    div_y[index >= 12] = 3

    return offset_x, offset_y, div_x, div_y


def rotation_flip_matrix_2d(
    rad: float, flip_x: bool = False, flip_y: bool = False
) -> Float[Tensor, "2 2"]:
    cos = math.cos(rad)
    sin = math.sin(rad)
    rot_mat = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32)
    flip_mat = torch.tensor(
        [
            [-1 if flip_x else 1, 0],
            [0, -1 if flip_y else 1],
        ],
        dtype=torch.float32,
    )

    return flip_mat @ rot_mat


def calculate_tangents(
    vertex_positions: Float[Tensor, "Nv 3"],
    vertex_normals: Float[Tensor, "Nv 3"],
    triangle_idxs: Integer[Tensor, "Nf 3"],
    face_uv: Float[Tensor, "Nf 3 2"],
) -> Float[Tensor, "Nf 3 4"]:  # noqa: F821
    vn_idx = [None] * 3
    pos = [None] * 3
    tex = face_uv.unbind(1)
    for i in range(0, 3):
        pos[i] = vertex_positions[triangle_idxs[:, i]]
        # t_nrm_idx is always the same as t_pos_idx
        vn_idx[i] = triangle_idxs[:, i]

    tangents = torch.zeros_like(vertex_normals)
    tansum = torch.zeros_like(vertex_normals)

    # Compute tangent space for each triangle
    duv1 = tex[1] - tex[0]
    duv2 = tex[2] - tex[0]
    dpos1 = pos[1] - pos[0]
    dpos2 = pos[2] - pos[0]

    tng_nom = dpos1 * duv2[..., 1:2] - dpos2 * duv1[..., 1:2]

    denom = duv1[..., 0:1] * duv2[..., 1:2] - duv1[..., 1:2] * duv2[..., 0:1]

    # Avoid division by zero for degenerated texture coordinates
    denom_safe = denom.clip(1e-6)
    tang = tng_nom / denom_safe

    # Update all 3 vertices
    for i in range(0, 3):
        idx = vn_idx[i][:, None].repeat(1, 3)
        tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(
            0, idx, torch.ones_like(tang)
        )  # tansum[n_i] = tansum[n_i] + 1
    # Also normalize it. Here we do not normalize the individual triangles first so larger area
    # triangles influence the tangent space more
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = F.normalize(tangents, dim=1)
    tangents = F.normalize(tangents - dot(tangents, vertex_normals) * vertex_normals)

    return tangents


def _rotate_uv_slices_consistent_space(
    vertex_positions: Float[Tensor, "Nv 3"],
    vertex_normals: Float[Tensor, "Nv 3"],
    triangle_idxs: Integer[Tensor, "Nf 3"],
    uv: Float[Tensor, "Nf 3 2"],
    index: Integer[Tensor, "Nf"],  # noqa: F821
):
    tangents = calculate_tangents(vertex_positions, vertex_normals, triangle_idxs, uv)
    pos_stack = torch.stack(
        [
            -vertex_positions[..., 1],
            vertex_positions[..., 0],
            torch.zeros_like(vertex_positions[..., 0]),
        ],
        dim=-1,
    )
    expected_tangents = F.normalize(
        torch.linalg.cross(
            vertex_normals, torch.linalg.cross(pos_stack, vertex_normals)
        ),
        -1,
    )

    actual_tangents = tangents[triangle_idxs]
    expected_tangents = expected_tangents[triangle_idxs]

    def rotation_matrix_2d(theta):
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.tensor([[c, -s], [s, c]])

    # Now find the rotation
    index_mod = index % 6  # Shouldn't happen. Just for safety
    for i in range(6):
        mask = index_mod == i
        if not mask.any():
            continue

        actual_mean_tangent = actual_tangents[mask].mean(dim=(0, 1))
        expected_mean_tangent = expected_tangents[mask].mean(dim=(0, 1))

        dot_product = torch.dot(actual_mean_tangent, expected_mean_tangent)
        cross_product = (
            actual_mean_tangent[0] * expected_mean_tangent[1]
            - actual_mean_tangent[1] * expected_mean_tangent[0]
        )
        angle = torch.atan2(cross_product, dot_product)

        rot_matrix = rotation_matrix_2d(angle).to(mask.device)
        # Center the uv coordinate to be in the range of -1 to 1 and 0 centered
        uv_cur = uv[mask] * 2 - 1  # Center it first
        # Rotate it
        uv[mask] = torch.einsum("ij,nfj->nfi", rot_matrix, uv_cur)

        # Rescale uv[mask] to be within the 0-1 range
        uv[mask] = (uv[mask] - uv[mask].min()) / (uv[mask].max() - uv[mask].min())

    return uv


def _handle_slice_uvs(
    uv: Float[Tensor, "Nf 3 2"],
    index: Integer[Tensor, "Nf"],  # noqa: F821
    island_padding: float,
    max_index: int = 6 * 2,
) -> Float[Tensor, "Nf 3 2"]:  # noqa: F821
    uc, vc = uv.unbind(-1)

    # Get the second slice (The first overlap)
    index_filter = [index == i for i in range(6, max_index)]

    # Normalize them to always fully fill the atlas patch
    for i, fi in enumerate(index_filter):
        if fi.sum() > 0:
            # Scale the slice but only up to a factor of 2
            # This keeps the texture resolution with the first slice in line (Half space in UV)
            uc[fi] = (uc[fi] - uc[fi].min()) / (uc[fi].max() - uc[fi].min()).clip(0.5)
            vc[fi] = (vc[fi] - vc[fi].min()) / (vc[fi].max() - vc[fi].min()).clip(0.5)

    uc_padded = (uc * (1 - 2 * island_padding) + island_padding).clip(0, 1)
    vc_padded = (vc * (1 - 2 * island_padding) + island_padding).clip(0, 1)

    return torch.stack([uc_padded, vc_padded], dim=-1)


def _handle_remaining_uvs(
    uv: Float[Tensor, "Nf 3 2"],
    index: Integer[Tensor, "Nf"],  # noqa: F821
    island_padding: float,
) -> Float[Tensor, "Nf 3 2"]:
    uc, vc = uv.unbind(-1)
    # Get all remaining elements
    remaining_filter = index >= 6 * 2
    squares_left = remaining_filter.sum()

    if squares_left == 0:
        return uv

    uc = uc[remaining_filter]
    vc = vc[remaining_filter]

    # Or remaining triangles are distributed in a rectangle
    # The rectangle takes 0.5 of the entire uv space in width and 1/3 in height
    ratio = 0.5 * (1 / 3)  # 1.5
    # sqrt(744/(0.5*(1/3)))

    mult = math.sqrt(squares_left / ratio)
    num_square_width = int(math.ceil(0.5 * mult))
    num_square_height = int(math.ceil(squares_left / num_square_width))

    width = 1 / num_square_width
    height = 1 / num_square_height

    # The idea is again to keep the texture resolution consistent with the first slice
    # This only occupys half the region in the texture chart but the scaling on the squares
    # assumes full coverage.
    clip_val = min(width, height) * 1.5
    # Now normalize the UVs with taking into account the maximum scaling
    uc = (uc - uc.min(dim=1, keepdim=True).values) / (
        uc.amax(dim=1, keepdim=True) - uc.amin(dim=1, keepdim=True)
    ).clip(clip_val)
    vc = (vc - vc.min(dim=1, keepdim=True).values) / (
        vc.amax(dim=1, keepdim=True) - vc.amin(dim=1, keepdim=True)
    ).clip(clip_val)
    # Add a small padding
    uc = (
        uc * (1 - island_padding * num_square_width * 0.5)
        + island_padding * num_square_width * 0.25
    ).clip(0, 1)
    vc = (
        vc * (1 - island_padding * num_square_height * 0.5)
        + island_padding * num_square_height * 0.25
    ).clip(0, 1)

    uc = uc * width
    vc = vc * height

    # And calculate offsets for each element
    idx = torch.arange(uc.shape[0], device=uc.device, dtype=torch.int32)
    x_idx = idx % num_square_width
    y_idx = idx // num_square_width
    # And move each triangle to its own spot
    uc = uc + x_idx[:, None] * width
    vc = vc + y_idx[:, None] * height

    uc = (uc * (1 - 2 * island_padding * 0.5) + island_padding * 0.5).clip(0, 1)
    vc = (vc * (1 - 2 * island_padding * 0.5) + island_padding * 0.5).clip(0, 1)

    uv[remaining_filter] = torch.stack([uc, vc], dim=-1)

    return uv


def _distribute_individual_uvs_in_atlas(
    face_uv: Float[Tensor, "Nf 3 2"],
    assigned_faces: Integer[Tensor, "Nf"],  # noqa: F821
    offset_x: Float[Tensor, "Nf"],  # noqa: F821
    offset_y: Float[Tensor, "Nf"],  # noqa: F821
    div_x: Float[Tensor, "Nf"],  # noqa: F821
    div_y: Float[Tensor, "Nf"],  # noqa: F821
    island_padding: float,
):
    # Place the slice first
    placed_uv = _handle_slice_uvs(face_uv, assigned_faces, island_padding)
    # Then handle the remaining overlap elements
    placed_uv = _handle_remaining_uvs(placed_uv, assigned_faces, island_padding)

    uc, vc = placed_uv.unbind(-1)
    uc = uc / div_x[:, None] + offset_x[:, None]
    vc = vc / div_y[:, None] + offset_y[:, None]

    uv = torch.stack([uc, vc], dim=-1).view(-1, 2)

    return uv


def _get_unique_face_uv(
    uv: Float[Tensor, "Nf 3 2"],
) -> Tuple[Float[Tensor, "Utex 3"], Integer[Tensor, "Nf"]]:  # noqa: F821
    unique_uv, unique_idx = torch.unique(uv, return_inverse=True, dim=0)
    # And add the face to uv index mapping
    vtex_idx = unique_idx.view(-1, 3)

    return unique_uv, vtex_idx


def _align_mesh_with_main_axis(
    vertex_positions: Float[Tensor, "Nv 3"], vertex_normals: Float[Tensor, "Nv 3"]
) -> Tuple[Float[Tensor, "Nv 3"], Float[Tensor, "Nv 3"]]:
    # Use pca to find the 2 main axis (third is derived by cross product)
    # Set the random seed so it's repeatable
    torch.manual_seed(0)
    _, _, v = torch.pca_lowrank(vertex_positions, q=2)
    main_axis, seconday_axis = v[:, 0], v[:, 1]

    main_axis: Float[Tensor, "3"] = F.normalize(main_axis, eps=1e-6, dim=-1)
    # Orthogonalize the second axis
    seconday_axis: Float[Tensor, "3"] = F.normalize(
        seconday_axis - dot(seconday_axis, main_axis) * main_axis, eps=1e-6, dim=-1
    )
    # Create perpendicular third axis
    third_axis: Float[Tensor, "3"] = F.normalize(
        torch.cross(main_axis, seconday_axis), dim=-1, eps=1e-6
    )

    # Check to which canonical axis each aligns
    main_axis_max_idx = main_axis.abs().argmax().item()
    seconday_axis_max_idx = seconday_axis.abs().argmax().item()
    third_axis_max_idx = third_axis.abs().argmax().item()

    # Now sort the axes based on the argmax so they align with thecanonoical axes
    # If two axes have the same argmax move one of them
    all_possible_axis = {0, 1, 2}
    cur_index = 1
    while len(set([main_axis_max_idx, seconday_axis_max_idx, third_axis_max_idx])) != 3:
        # Find missing axis
        missing_axis = all_possible_axis - set(
            [main_axis_max_idx, seconday_axis_max_idx, third_axis_max_idx]
        )
        missing_axis = missing_axis.pop()
        # Just assign it to third axis as it had the smallest contribution to the
        # overall shape
        if cur_index == 1:
            third_axis_max_idx = missing_axis
        elif cur_index == 2:
            seconday_axis_max_idx = missing_axis
        else:
            raise ValueError("Could not find 3 unique axis")
        cur_index += 1

    if len({main_axis_max_idx, seconday_axis_max_idx, third_axis_max_idx}) != 3:
        raise ValueError("Could not find 3 unique axis")

    axes = [None] * 3
    axes[main_axis_max_idx] = main_axis
    axes[seconday_axis_max_idx] = seconday_axis
    axes[third_axis_max_idx] = third_axis
    # Create rotation matrix from the individual axes
    rot_mat = torch.stack(axes, dim=1).T

    # Now rotate the vertex positions and vertex normals so the mesh aligns with the main axis
    vertex_positions = torch.einsum("ij,nj->ni", rot_mat, vertex_positions)
    vertex_normals = torch.einsum("ij,nj->ni", rot_mat, vertex_normals)

    return vertex_positions, vertex_normals


def box_projection_uv_unwrap(
    vertex_positions: Float[Tensor, "Nv 3"],
    vertex_normals: Float[Tensor, "Nv 3"],
    triangle_idxs: Integer[Tensor, "Nf 3"],
    island_padding: float,
) -> Tuple[Float[Tensor, "Utex 3"], Integer[Tensor, "Nf"]]:  # noqa: F821
    # Align the mesh with main axis directions first
    vertex_positions, vertex_normals = _align_mesh_with_main_axis(
        vertex_positions, vertex_normals
    )

    bbox: Float[Tensor, "2 3"] = torch.stack(
        [vertex_positions.min(dim=0).values, vertex_positions.max(dim=0).values], dim=0
    )
    # First decide in which cube face the triangle is placed
    face_uv, face_index = _box_assign_vertex_to_cube_face(
        vertex_positions, vertex_normals, triangle_idxs, bbox
    )

    # Rotate the UV islands in a way that they align with the radial z tangent space
    face_uv = _rotate_uv_slices_consistent_space(
        vertex_positions, vertex_normals, triangle_idxs, face_uv, face_index
    )

    # Then find where where the face is placed in the atlas.
    # This has to detect potential overlaps
    assigned_atlas_index = _assign_faces_uv_to_atlas_index(
        vertex_positions, triangle_idxs, face_uv, face_index
    )

    # Then figure out the final place in the atlas based on the assignment
    offset_x, offset_y, div_x, div_y = _find_slice_offset_and_scale(
        assigned_atlas_index
    )

    # Next distribute the faces in the uv atlas
    placed_uv = _distribute_individual_uvs_in_atlas(
        face_uv, assigned_atlas_index, offset_x, offset_y, div_x, div_y, island_padding
    )

    # And get the unique per-triangle UV coordinates
    return _get_unique_face_uv(placed_uv)
