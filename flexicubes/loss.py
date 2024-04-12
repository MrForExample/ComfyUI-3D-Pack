# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch
import torch_scatter

###############################################################################
# Pytorch implementation of the developability regularizer introduced in paper 
# "Developability of Triangle Meshes" by Stein et al.
###############################################################################
def mesh_developable_reg(mesh):

    verts = mesh.vertices
    tris = mesh.faces

    device = verts.device
    V = verts.shape[0]
    F = tris.shape[0]

    POS_EPS = 1e-6
    REL_EPS = 1e-6

    def normalize(vecs):
        return vecs / (torch.linalg.norm(vecs, dim=-1, keepdim=True) + POS_EPS)

    tri_pos = verts[tris]

    vert_normal_covariance_sum = torch.zeros((V, 9), device=device)
    vert_area = torch.zeros(V, device=device)
    vert_degree = torch.zeros(V, dtype=torch.int32, device=device)

    for iC in range(3):  # loop over three corners of each triangle

        # gather tri verts
        pRoot = tri_pos[:, iC, :]
        pA = tri_pos[:, (iC + 1) % 3, :]
        pB = tri_pos[:, (iC + 2) % 3, :]

        # compute the corner angle & normal
        vA = pA - pRoot
        vAn = normalize(vA)
        vB = pB - pRoot
        vBn = normalize(vB)
        area_normal = torch.linalg.cross(vA, vB, dim=-1)
        face_area = 0.5 * torch.linalg.norm(area_normal, dim=-1)
        normal = normalize(area_normal)
        corner_angle = torch.acos(torch.clamp(torch.sum(vAn * vBn, dim=-1), min=-1., max=1.))

        # add up the contribution to the covariance matrix
        outer = normal[:, :, None] @ normal[:, None, :]
        contrib = corner_angle[:, None] * outer.reshape(-1, 9)

        # scatter the result to the appropriate matrices
        vert_normal_covariance_sum = torch_scatter.scatter_add(src=contrib,
                                                               index=tris[:, iC],
                                                               dim=-2,
                                                               out=vert_normal_covariance_sum)

        vert_area = torch_scatter.scatter_add(src=face_area / 3.,
                                              index=tris[:, iC],
                                              dim=-1,
                                              out=vert_area)

        vert_degree = torch_scatter.scatter_add(src=torch.ones(F, dtype=torch.int32, device=device),
                                                index=tris[:, iC],
                                                dim=-1,
                                                out=vert_degree)

    # The energy is the smallest eigenvalue of the outer-product matrix
    vert_normal_covariance_sum = vert_normal_covariance_sum.reshape(
        -1, 3, 3)  # reshape to a batch of matrices
    vert_normal_covariance_sum = vert_normal_covariance_sum + torch.eye(
        3, device=device)[None, :, :] * REL_EPS

    min_eigvals = torch.min(torch.linalg.eigvals(vert_normal_covariance_sum).abs(), dim=-1).values

    # Mask out degree-3 vertices
    vert_area = torch.where(vert_degree == 3, torch.tensor(0, dtype=vert_area.dtype,device=vert_area.device), vert_area)

    # Adjust the vertex area weighting so it is unit-less, and 1 on average
    vert_area = vert_area * (V / torch.sum(vert_area, dim=-1, keepdim=True))

    return vert_area * min_eigvals 

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff