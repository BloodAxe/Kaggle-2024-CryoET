# 1st place CryoET challenge solution

## Getting started

```bash
pip install -r requirements.txt
```

## Training your own models

The train script tailored for 4x3090 GPUs. The training script is `train.sh`. You can run it with:

```bash
bash train.sh
```

If you have different number of GPUs you may still train but update `--nproc-per-node=4` with number of GPUs you have. 
If you have different GPUs with different amount of VRAM you can scale the batch size accordingly. 

## Exporting to ONNX

COMING SOON


## Inference kernel

COMING SOON