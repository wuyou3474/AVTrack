# AVTrack
The official implementation for the **ICML 2024** paper [ [**Learning Adaptive and View-Invariant Vision Transformer for Real-Time UAV Tracking**](https://openreview.net/pdf?id=eaNLvrP8n1)]

[Models & Raw Results](https://pan.baidu.com/s/1sAGbZxIHfw0ahdpKVeqhIg?pwd=avtr) Baidu Driver: avtr [Models & Raw Results](https://drive.google.com/drive/folders/1dUi_7bSEpYgMQOPCvPZG7FKDyFz9oLh0?usp=sharing) Google Driver

##  Methodology

<p align="center">
  <img width="85%" src="assets/AVTrack.png" alt="AVTrack"/>
</p>

## Usage
### Installation
Create and activate a conda environment:
```
conda create -n AVTrack python=3.8
conda activate AVTrack
```

Install the required packages:
```
pip install -r requirement.txt
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_AVTrack>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Download pre-trained [DeiT-Tiny weights](https://github.com/facebookresearch/deit), [Eva02-Tiny weights](https://github.com/baaivision/EVA/tree/master/EVA-02) , and [ViT-Tiny weights](https://github.com/google-research/vision_transformer)  and put it under `$USER_ROOT$//home/lsw/.cache/torch/hub/checkpoints/. 
```
python tracking/train.py --script avtrack --config deit_tiny_patch16_224 --save_dir ./output --mode single
```


### Testing
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1dUi_7bSEpYgMQOPCvPZG7FKDyFz9oLh0?usp=sharing) or [BaiduNetDisk](https://pan.baidu.com/s/1sAGbZxIHfw0ahdpKVeqhIg?pwd=avtr)

Put the downloaded weights on `<PATH_of_AVTrack>/output/checkpoints/train/avtrack/deit_tiny_patch16_224`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

 Testing examples:
- UAVDT
```
python tracking/test.py avtrack deit_tiny_patch16_224 --dataset uavdt --threads 4 --num_gpus 1
python tracking/analysis_results.py # need to modify tracker configs and names
```

### Test FLOPs
```
# Profiling AVTrack
python tracking/profile_model.py --script avtrack --config deit_tiny_patch16_224
```


## Acknowledgment
* This repo is based on [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library which are excellent works and help us to quickly implement our ideas.

* We use the implementation of the DeiT, Eva02, and ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repo. 

## Presentation Demo

[![Demo](assets/presentation.png)](https://www.youtube.com/watch?v=N5nW1pWblZw)


## Citation
If our work is useful for your research, please consider citing:
```Bibtex
@inproceedings{lilearning,
  title={Learning Adaptive and View-Invariant Vision Transformer for Real-Time UAV Tracking},
  author={Li, Yongxin and Liu, Mengyuan and Wu, You and Wang, Xucheng and Yang, Xiangyang and Li, Shuiwang},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
