# Dataset Preparation
We provide the data preprocessing pipeline for PartCrafter. By following the instructions, you can generate the training data from the raw GLB data. While we are considering releasing the preprocessed dataset, please note that it may take some time before it becomes available. 

## Download Raw Data
Our final model uses a subset of [Objaverse](https://huggingface.co/datasets/allenai/objaverse) provided by [LGM](https://github.com/ashawkey/objaverse_filter) and [Amazon Berkeley Objects (ABO) Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html). Please download the raw GLB files according to their instructions. You can also use other source of data. 

## Data Preprocess
We provide several scripts to preprocess the raw GLB files [here](./preprocess/). These scripts are minimal implementations and illustrate the whole preprocessing pipeline on a single 3D object. 

1. Sample points from mesh surface
```
python datasets/preprocess/mesh_to_point.py --input assets/objects/scissors.glb --output preprocessed_data
```

2. Render images
```
python datasets/preprocess/render.py --input assets/objects/scissors.glb --output preprocessed_data
```

3. Remove background for rendered images and resize to 90%
```
python datasets/preprocess/rmbg.py --input preprocessed_data/scissors/rendering.png --output preprocessed_data
```

4. (Optional) Calculate IoU
```
python datasets/preprocess/calculate_iou.py --input assets/objects/scissors.glb --output preprocessed_data
```
After preprocessing, you can generate a dataset configuration file according to the example configuration file with your own data path. 

To preprocess a folder of meshes, run
```
python datasets/preprocess/preprocess.py --input assets/objects --output preprocessed_data
```
This will also generate a configuration file in `./preprocessed_data/object_part_configs.json`. 

## Dataset Configuration
The training code requires specific format of dataset configuration. I provide an example configuration [here](example_configs.json). You can use it as a template to configure your own dataset. A minimal legal configuration file should be like: 

```
[
    {
        "mesh_path": "/path/to/object.glb",
        "surface_path": "/path/to/object.npy",
        "image_path": "/path/to/object.png",
        "num_parts": 4,
        "iou_mean": 0.5,
        "iou_max": 0.9,
        "valid": true
    }, 
    {
        ...
    },
    ...
]
```
Explaination:
- `mesh_path`: The path to the GLB file of the object.
- `surface_path`: The path to the npy file of the object surface points.
- `image_path`: The path to the rendered image of the object (after removing background).
- `num_parts`: The number of parts of the object.
- `iou_mean`: The mean IoU of the object parts.
- `iou_max`: The max IoU of the object parts.
- `valid`: Whether the object is valid. If set to false, the object will be filtered out during training.