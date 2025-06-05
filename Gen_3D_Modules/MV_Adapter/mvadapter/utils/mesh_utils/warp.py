import os

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from PIL import Image, ImageDraw
from tqdm import tqdm


def draw(img, lines):
    """
    lines: n x 4, x1,y2,x2,y2
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if isinstance(img, np.ndarray):
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255.0).astype(np.uint8)
        img = Image.fromarray(img)

    img = img.resize((512, 512), Image.Resampling.BICUBIC)
    w, h = img.size
    assert w == h
    if isinstance(lines, torch.Tensor):
        lines = lines.detach().cpu().numpy()
    lines = (lines + 1) / 2 * w

    img1 = ImageDraw.Draw(img)
    for line in lines:
        img1.line(line.tolist(), fill="red", width=1)
    return img


def construct_grid_mesh(n_grid):
    vertices = []
    vertices_movable_ids = []
    idx = 0
    for j in range(n_grid + 1):
        for i in range(n_grid + 1):
            # 3D fit in trimesh format
            if 0 < i < n_grid and 0 < j < n_grid:
                vertices_movable_ids.append(idx)
            vertices.append([i / n_grid, j / n_grid, 1.0 / 2])
            idx += 1

    vertices = np.array(vertices)
    vertices = 2 * vertices - 1

    faces = []
    for j in range(n_grid):
        for i in range(n_grid):
            # clockwise
            faces.append(
                [
                    i + j * (n_grid + 1),
                    i + 1 + j * (n_grid + 1),
                    i + (j + 1) * (n_grid + 1),
                ]
            )
            faces.append(
                [
                    i + 1 + j * (n_grid + 1),
                    i + 1 + (j + 1) * (n_grid + 1),
                    i + (j + 1) * (n_grid + 1),
                ]
            )
    faces = np.array(faces)
    vertices_movable_ids = np.array(vertices_movable_ids)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh, vertices_movable_ids, vertices


def compute_warp_field(
    ctx,
    src_images_tensor,
    tgt_images_tensor,
    n_grid,
    optim_res,
    optim_step_per_res,
    lambda_reg,
    temp_dir,
    verbose,
    device,
):
    """
    src_images_tensor: 4 H W 4
    tgt_images_tensor: 4 H W 4
    return: 4 H W 4
    """
    lam_reg = lambda_reg
    lam_mask = 2
    mesh, _, vertices_np = construct_grid_mesh(n_grid)

    # vertices = mesh.vertices
    faces = mesh.faces
    edges = mesh.edges_unique

    n_degree = torch.tensor(mesh.vertex_degree)
    vertices = torch.tensor(vertices_np, device=device, dtype=torch.float32)
    vertices_unopt = vertices.clone()
    faces = torch.tensor(faces, device=device)
    edges = torch.tensor(edges, device=device)

    warped_images_tensor = []
    bs = src_images_tensor.shape[0]

    for img_idx in range(bs):
        src_image_tensor = src_images_tensor[img_idx]
        tgt_image_tensor = tgt_images_tensor[img_idx]

        if verbose and temp_dir is not None:
            vis_dir = os.path.join(temp_dir, f"{img_idx}")
            os.makedirs(vis_dir, exist_ok=True)
        # prepare for nvdiff rendering
        v_origin_homo = torch.cat(
            [vertices_unopt, torch.ones_like(vertices_unopt[..., :1])], dim=-1
        )

        # prepare parameters
        v_move_indices = torch.where(n_degree == 6)[0]
        vertices_movable = vertices[v_move_indices].detach().clone()
        vertices_movable.requires_grad = True
        opt = torch.optim.Adam([vertices_movable], lr=0.02)

        for resl in optim_res:
            rast, _ = dr.rasterize(
                ctx,
                v_origin_homo[None, ...].float(),
                faces.int(),
                (resl, resl),
                grad_db=True,
            )
            face_ids = rast[..., 3].long() - 1
            assert torch.all(face_ids >= 0)
            u = rast[..., 0]
            v = rast[..., 1]
            pixel_vertice_ids = faces[face_ids]
            pixel_bary_coords = torch.stack([u, v, 1 - u - v], dim=-1)

            src_img = F.interpolate(
                rearrange(src_image_tensor[..., :3], "H W C -> () C H W"),
                size=(resl, resl),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            tgt_img = F.interpolate(
                rearrange(tgt_image_tensor[..., :3], "H W C -> () C H W"),
                size=(resl, resl),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # src_img, tgt_img: 1 C H W
            if verbose:
                Image.fromarray(
                    (tgt_img[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(
                        np.uint8
                    )
                ).save(os.path.join(vis_dir, f"target_{resl:04d}.png"))
            with tqdm(range(optim_step_per_res), disable=not verbose) as pbar:
                for i in pbar:
                    opt.zero_grad()
                    vertices_all = vertices_unopt.detach().clone()
                    vertices_all[v_move_indices] = vertices_movable

                    pixel_vertices = vertices_all[
                        pixel_vertice_ids
                    ]  # 1 x 512 x 512 x 3 x 3
                    pixel_coords_inter = torch.sum(
                        pixel_vertices * pixel_bary_coords[..., None], dim=-2
                    )
                    src_img_warped = F.grid_sample(
                        src_img,
                        pixel_coords_inter[..., :2],
                        mode="bilinear",
                        align_corners=False,
                    )

                    edge_vertices_all = vertices_all[edges]
                    edge_vertices_unopt = vertices_unopt[edges]

                    edge_len_all = torch.linalg.norm(
                        edge_vertices_all[:, 0, :2] - edge_vertices_all[:, 1, :2],
                        dim=-1,
                    )
                    edge_len_unopt = torch.linalg.norm(
                        edge_vertices_unopt[:, 0, :2] - edge_vertices_all[:, 1, :2],
                        dim=-1,
                    )

                    reg_loss = ((edge_len_all - edge_len_unopt) ** 2).mean()

                    # mask_loss = ((src_img_warped[:, 3:4, :, :] - tgt_img[:, 3:4, :, :])**2).mean()
                    # rgb_loss = (((src_img_warped[:, :3, :, :] - tgt_img[:, :3, :, :]) * tgt_img[:, 3:4, :, :])**2).mean()
                    # loss = rgb_loss + lam_reg * reg_loss + lam_mask * mask_loss
                    img_loss = ((src_img_warped - tgt_img) ** 2).mean()
                    loss = img_loss + lam_reg * reg_loss

                    loss.backward()
                    opt.step()
                    if verbose:
                        print(img_loss.item(), reg_loss.item())

                    if verbose:
                        draw(
                            src_img[0].permute(1, 2, 0),
                            edge_vertices_all[:, :, :2].reshape(-1, 4),
                        ).save(os.path.join(vis_dir, f"src_{resl:04d}_{i:03d}.png"))
                        Image.fromarray(
                            (
                                torch.cat(
                                    [
                                        tgt_img,
                                        src_img_warped,
                                        (tgt_img - src_img_warped).abs(),
                                    ],
                                    dim=-1,
                                )[0]
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                                * 255.0
                            ).astype(np.uint8)
                        ).save(os.path.join(vis_dir, f"opt_{resl:04d}_{i:03d}.png"))
                        warped_this_resl = Image.fromarray(
                            (
                                src_img_warped[0]
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                                * 255.0
                            ).astype(np.uint8)
                        )
                        warped_this_resl.save(
                            os.path.join(vis_dir, f"warped_{resl:04d}_{i:03d}.png")
                        )

        # warp on 512 resolution
        resl = src_image_tensor.shape[1]
        with torch.no_grad():
            rast, _ = dr.rasterize(
                ctx,
                v_origin_homo[None, ...].float(),
                faces.int(),
                (resl, resl),
                grad_db=True,
            )
            face_ids = rast[..., 3].long() - 1
            assert torch.all(face_ids >= 0)
            u = rast[..., 0]
            v = rast[..., 1]
            pixel_vertice_ids = faces[face_ids]
            pixel_bary_coords = torch.stack([u, v, 1 - u - v], dim=-1)

            vertices_all = vertices_unopt.detach().clone()
            vertices_all[v_move_indices] = vertices_movable

            pixel_vertices = vertices_all[pixel_vertice_ids]  # 1 x 512 x 512 x 3 x 3
            pixel_coords_inter = torch.sum(
                pixel_vertices * pixel_bary_coords[..., None], dim=-2
            )
            src_img_warped = rearrange(
                F.grid_sample(
                    rearrange(src_image_tensor, "H W C -> () C H W"),
                    pixel_coords_inter[..., :2],
                    mode="bicubic",
                    align_corners=False,
                ),
                "() C H W -> H W C",
            ).clamp(0, 1)
        warped_images_tensor.append(src_img_warped)

    warped_images_tensor = torch.stack(warped_images_tensor, dim=0)  # 4 H W 3

    return warped_images_tensor
