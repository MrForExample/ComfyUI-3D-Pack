# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import requests
from zipfile import ZipFile
from tqdm import tqdm
import os

def download_file(url, output_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception("ERROR, something went wrong")


url = "https://vcg.isti.cnr.it/Publications/2014/MPZ14/inputmodels.zip"
zip_file_path = './data/inputmodels.zip'

os.makedirs('./data', exist_ok=True)

download_file(url, zip_file_path)

with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('./data')

os.remove(zip_file_path)

print("Download and extraction complete.")
