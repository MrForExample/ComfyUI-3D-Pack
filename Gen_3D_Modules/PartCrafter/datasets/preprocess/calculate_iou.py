import os
import trimesh
import numpy as np
import argparse
import json

from partcrafter_src.utils.data_utils import normalize_mesh
from partcrafter_src.utils.metric_utils import compute_IoU_for_scene

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='assets/objects/scissors.glb')
    parser.add_argument('--output', type=str, default='preprocessed_data')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_path, mesh_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    config = {
        'iou_mean': 0.0,
        'iou_max': 0.0,
        'iou_list': [],
    }
    mesh = normalize_mesh(trimesh.load(input_path, process=False))
    try:
        iou_list = compute_IoU_for_scene(mesh, return_type='iou_list')
        config['iou_list'] = iou_list
        config['iou_mean'] = np.mean(iou_list)
        config['iou_max'] = np.max(iou_list)
    except:
        config['iou_list'] = []
        config['iou_mean'] = 0.0
        config['iou_max'] = 0.0

    json.dump(config, open(os.path.join(output_path, f'iou.json'), 'w'), indent=4)