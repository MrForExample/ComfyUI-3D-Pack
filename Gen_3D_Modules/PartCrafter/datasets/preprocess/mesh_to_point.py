import os
import trimesh
import numpy as np
import argparse
import json

from partcrafter_src.utils.data_utils import scene_to_parts, mesh_to_surface, normalize_mesh

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
        "num_parts": 0, 
    }

    # sample points from mesh surface
    mesh = trimesh.load(input_path, process=False)
    mesh = normalize_mesh(mesh)
    config["num_parts"] = len(mesh.geometry)
    if config["num_parts"] > 1 and config["num_parts"] <= 16:
        parts = scene_to_parts(
            mesh,
            return_type="point",
            normalize=False
        )
    else:
        parts = []
    mesh = mesh.to_geometry()
    object = mesh_to_surface(mesh, return_dict=True)
    datas = {
        "object": object,
        "parts": parts,
    }
    # save points
    np.save(os.path.join(output_path, 'points.npy'), datas)

    # save config
    with open(os.path.join(output_path, 'num_parts.json'), 'w') as f:
        json.dump(config, f, indent=4)