# 1st place CryoET challenge solution

This repo contains training code for OD part of our first place solution for CryoET challenge.
Please check out solution writeups for more details:
* https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561440#3117046
* https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/561510#3116985

## Getting started

First things first - clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

If you want to use pretrained weights for SegResNet - download them from the following link: https://github.com/Project-MONAI/model-zoo/releases/download/hosting_storage_v1/wholeBody_ct_segmentation_v0.1.9.zip.
Then, unzip the model checkpoint to `pretrained/wholeBody_ct_segmentation/models/model.pt`.
If you don't want to use pretrained weights, simply remove the line `--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \` from train script. 

## Training your own models

The train script tailored for 4x3090 GPUs. The training script is `train.sh`. You can run it with:

```bash
bash train.sh
```

If you have different number of GPUs you may still train but update `--nproc-per-node=4` with number of GPUs you have. 
If you have different GPUs with different amount of VRAM you can scale the batch size accordingly. 

## Exporting to ONNX

Once you have trained your models, to build a single ONNX ensemble, use the following command:

```bash 
python export_ensemble_onnx.py --output_onnx=ensemble.onnx --batch_size=1 <AS MANY CHECKPOINTS AS YOU WANT>
```

## Converting ONNX to TensorRT

An example of inference kernel is provided in [scripts/kaggle_convert_onnx_to_tensorrt.py](scripts/kaggle_convert_onnx_to_tensorrt.py). 

## Inference kernel

An example of inference kernel is provided in [scripts/kaggle_inference_with_iobinding.py](scripts/kaggle_inference_with_iobinding.py). 

