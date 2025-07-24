import os
import json
import argparse
import time
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='assets/objects')
    parser.add_argument('--output', type=str, default='preprocessed_data')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    assert os.path.exists(input_path), f'{input_path} does not exist'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for mesh_name in tqdm(os.listdir(input_path)):
        mesh_path = os.path.join(input_path, mesh_name)
        # 1. Sample points from mesh surface
        os.system(f"python datasets/preprocess/mesh_to_point.py --input {mesh_path} --output {output_path}")
        # 2. Render images
        os.system(f"python datasets/preprocess/render.py --input {mesh_path} --output {output_path}")
        # 3. Remove background for rendered images and resize to 90%
        export_mesh_folder = os.path.join(output_path, mesh_name.replace('.glb', ''))
        export_rendering_path = os.path.join(export_mesh_folder, 'rendering.png')
        os.system(f"python datasets/preprocess/rmbg.py --input {export_rendering_path} --output {output_path}")
        # 4. (Optional) Calculate IoU
        os.system(f"python datasets/preprocess/calculate_iou.py --input {mesh_path} --output {output_path}")
        time.sleep(1)
    
    # generate configs
    configs = []
    for mesh_name in tqdm(os.listdir(input_path)):
        mesh_path = os.path.join(output_path, mesh_name.replace('.glb', ''))
        num_parts_path = os.path.join(mesh_path, 'num_parts.json')
        surface_path = os.path.join(mesh_path, 'points.npy')
        image_path = os.path.join(mesh_path, 'rendering_rmbg.png')
        iou_path = os.path.join(mesh_path, 'iou.json')
        config = {
            "file": mesh_name,
            "num_parts": 0,
            "valid": False,
            "mesh_path": os.path.join(input_path, mesh_name),
            "surface_path": None,
            "image_path": None,
            "iou_mean": 0.0,
            "iou_max": 0.0
        }
        try:
            config["num_parts"] = json.load(open(num_parts_path))['num_parts']
            iou_config = json.load(open(iou_path))
            config['iou_mean'] = iou_config['iou_mean']
            config['iou_max'] = iou_config['iou_max']
            assert os.path.exists(surface_path)
            config['surface_path'] = surface_path
            assert os.path.exists(image_path)
            config['image_path'] = image_path
            config['valid'] = True
            configs.append(config)
        except:
            continue
    
    configs_path = os.path.join(output_path, 'object_part_configs.json')
    json.dump(configs, open(configs_path, 'w'), indent=4)