import os
import trimesh
import numpy as np
import argparse
import json
import torch
from huggingface_hub import snapshot_download

from partcrafter_src.utils.image_utils import prepare_image
from partcrafter_src.models.briarmbg import BriaRMBG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='preprocessed_data/scissors/scissors.png')
    parser.add_argument('--output', type=str, default='preprocessed_data')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    assert os.path.exists(input_path), f'{input_path} does not exist'

    mesh_name = os.path.basename(os.path.dirname(input_path))
    output_path = os.path.join(output_path, mesh_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rendering_rmbg = prepare_image(input_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net, device=device)
    rendering_rmbg.save(os.path.join(output_path, f'rendering_rmbg.png'))