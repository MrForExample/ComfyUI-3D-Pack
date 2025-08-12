from partcrafter_src.utils.typing_utils import *

import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors

def sample_from_mesh(
    mesh: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
):
    if num_samples is None:
        return mesh.vertices
    else:
        return mesh.sample(num_samples)

def sample_two_meshes(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
):
    points1 = sample_from_mesh(mesh1, num_samples)
    points2 = sample_from_mesh(mesh2, num_samples)
    return points1, points2

def compute_nearest_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    metric: str = 'l2'
) -> np.ndarray:
    # Compute nearest neighbor distance from points1 to points2
    nn = NearestNeighbors(n_neighbors=1, leaf_size=30, algorithm='kd_tree', metric=metric).fit(points2)
    min_dist = nn.kneighbors(points1)[0]
    return min_dist

def compute_mutual_nearest_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    metric: str = 'l2'
) -> np.ndarray:
    min_1_to_2 = compute_nearest_distance(points1, points2, metric=metric)
    min_2_to_1 = compute_nearest_distance(points2, points1, metric=metric)
    return min_1_to_2, min_2_to_1

def compute_mutual_nearest_distance_for_meshes(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
    metric: str = 'l2'
) -> Tuple[np.ndarray, np.ndarray]:
    points1 = sample_from_mesh(mesh1, num_samples)
    points2 = sample_from_mesh(mesh2, num_samples)
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance(points1, points2, metric=metric)
    return min_1_to_2, min_2_to_1

def compute_chamfer_distance(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: int = 10000,
    metric: str = 'l2'
):
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
    chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
    return chamfer_dist

def compute_f_score(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: int = 10000,
    threshold: float = 0.1,
    metric: str = 'l2'
):
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
    precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
    precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    return fscore

def compute_cd_and_f_score(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_samples: Optional[int] = 10000,
    threshold: float = 0.1,
    metric: str = 'l2'
):
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance_for_meshes(mesh1, mesh2, num_samples, metric=metric)
    chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
    precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
    precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    return chamfer_dist, fscore

def compute_cd_and_f_score_in_training(
    gt_surface: np.ndarray,
    pred_mesh: trimesh.Trimesh,
    num_samples: int = 204800,
    threshold: float = 0.1,
    metric: str = 'l2'
):
    gt_points = gt_surface[:, :3]
    num_samples = max(num_samples, gt_points.shape[0])
    gt_points = gt_points[np.random.choice(gt_points.shape[0], num_samples, replace=False)]
    pred_points = sample_from_mesh(pred_mesh, num_samples)
    min_1_to_2, min_2_to_1 = compute_mutual_nearest_distance(gt_points, pred_points, metric=metric)
    chamfer_dist = np.mean(min_2_to_1) + np.mean(min_1_to_2)
    precision_1 = np.mean((min_1_to_2 < threshold).astype(np.float32))
    precision_2 = np.mean((min_2_to_1 < threshold).astype(np.float32))
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    return chamfer_dist, fscore

def get_voxel_set(
    mesh: trimesh.Trimesh,
    num_grids: int = 64,
    scale: float = 2.0,
):
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("mesh must be a trimesh.Trimesh object")
    pitch = scale / num_grids
    voxel_girds: trimesh.voxel.base.VoxelGrid = mesh.voxelized(pitch=pitch).fill()
    voxels = set(map(tuple, np.round(voxel_girds.points / pitch).astype(int)))
    return voxels

def compute_IoU(
    mesh1: trimesh.Trimesh,
    mesh2: trimesh.Trimesh,
    num_grids: int = 64,
    scale: float = 2.0,
):
    if not isinstance(mesh1, trimesh.Trimesh) or not isinstance(mesh2, trimesh.Trimesh):
        raise ValueError("mesh1 and mesh2 must be trimesh.Trimesh objects")
    voxels1 = get_voxel_set(mesh1, num_grids, scale)
    voxels2 = get_voxel_set(mesh2, num_grids, scale)
    intersection = voxels1 & voxels2
    union = voxels1 | voxels2
    iou = len(intersection) / len(union) if len(union) > 0 else 0.0
    return iou

def compute_IoU_for_scene(
    scene: Union[trimesh.Scene, List[trimesh.Trimesh]],
    num_grids: int = 64,
    scale: float = 2.0,
    return_type: Literal["iou", "iou_list"] = "iou",
):
    if isinstance(scene, trimesh.Scene):
        scene = scene.dump()
    if isinstance(scene, list) and len(scene) > 1 and isinstance(scene[0], trimesh.Trimesh):
        meshes = scene
    else:
        raise ValueError("scene must be a trimesh.Scene object or a list of trimesh.Trimesh objects")
    ious = []
    for i in range(len(meshes)):
        for j in range(i+1, len(meshes)):
            iou = compute_IoU(meshes[i], meshes[j], num_grids, scale)
            ious.append(iou)
    if return_type == "iou":
        return np.mean(ious)
    elif return_type == "iou_list":
        return ious
    else:
        raise ValueError("return_type must be 'iou' or 'iou_list'")