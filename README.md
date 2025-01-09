# Baseline SegResNet

| Fold | Score  | Threshold  |
|------|--------|------------|
| 0    | 0.7810 | 0.0625     |
| 1    | 0.7634 | 0.0625     |
| 2    | 0.7191 | 0.0625     |
| 3    | 0.7980 | 0.0900     |
| 4    | 0.8135 | 0.0900     |

CV: 0.775
LB: 0.736 (Score 0.1, TopK 2048 per class)

# Baseline Detection SegResNet

| Fold | Score  | Threshold |
|------|--------|-----------|
| 0    | 0.8190 | 0.1945    |
| 1    | 0.7984 | 0.1945    |
| 2    | 0.7443 | 0.1584    |
| 3    | 0.8271 | 0.1584    |
| 4    | 0.8264 | 0.1584    |

CV: 0.803 (Mean threshold 0.17278)
Do not fit into time limit

# Baseline SegResNetV2

| Fold | Score  | Apo-Ferritin Threshold | Beta-Galactosidase Threshold | RBSM Threshold | TRGLB Threshold | VLP Threshold |
|------|--------|------------------------|------------------------------|----------------|-----------------|---------------|
| 0    | 0.8272 | 0.1945                 | 0.23                         | 0.1945         | 0.126           | 0.488         | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_0_rc_ic_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_0_1802-score-0.8272-at-0.903-0.700-0.901-0.793-1.000.ckpt
| 1    | 0.8004 | 0.43                   | 0.1945                       | 0.27           | 0.27            | 0.87          | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_1_rc_ic_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_1_1166-score-0.8004-at-0.945-0.701-0.858-0.763-0.873.ckpt
| 2    | 0.7906 | 0.23                   | 0.27                         | 0.27           | 0.27            | 0.68          | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_2_rc_ic_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_2_636-score-0.7906-at-0.962-0.675-0.814-0.758-0.893.ckpt
| 3    | 0.8296 | 0.1945                 | 0.23                         | 0.23           | 0.23            | 0.55          | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_3_rc_ic_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_3_2332-score-0.8296-at-0.953-0.768-0.865-0.729-0.996.ckpt
| 4    | 0.8121 | 0.1584                 | 0.1945                       | 0.23           | 0.1585          | 0.403         | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_4_rc_ic_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_4_1378-score-0.8121-at-0.941-0.676-0.872-0.761-0.999.ckpt

CV: 0.81198

LB: [0.20, 0.20, 0.20, 0.20, 0.5] -> 0.739
LB: [0.20, 0.20, 0.20, 0.20, 0.6] -> 0.727
LB: [0.20, 0.20, 0.20, 0.20, 0.7] -> 0.710
LB: [0.15, 0.20, 0.20, 0.20, 0.5] -> 0.739
LB: [0.25, 0.20, 0.20, 0.20, 0.5] -> 0.738
LB: [0.20, 0.20, 0.20, 0.20, 0.4] -> 0.745

# Baseline SegResNetV2 Flip Augs & Slight rotation along Y & X

| Fold | Score  | AFRT  | BGT   | RBSM  | TRGLB | VLP   |
|------|--------|-------|-------|-------|-------|-------|
| 0    | 0.8466 | 0.234 | 0.325 | 0.278 | 0.234 | 0.759 | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_0_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_0_2862-score-0.8466-at-0.234-0.325-0.278-0.234-0.759.ckpt
| 1    | 0.8177 | 0.278 | 0.278 | 0.234 | 0.234 | 0.325 | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_1_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_1_1590-score-0.8177-at-0.278-0.278-0.234-0.234-0.325.ckpt 
| 2    | 0.7953 | 0.430 | 0.234 | 0.278 | 0.234 | 0.489 | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_2_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_2_1060-score-0.7953-at-0.430-0.234-0.278-0.234-0.489.ckpt 
| 3    | 0.8234 | 0.158 | 0.194 | 0.278 | 0.278 | 0.158 | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_3_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_3_2438-score-0.8234-at-0.158-0.194-0.278-0.278-0.158.ckpt 
| 4    | 0.8391 | 0.234 | 0.194 | 0.194 | 0.234 | 0.278 | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_4_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_4_2226-score-0.8391-at-0.234-0.194-0.194-0.234-0.278.ckpt 
| Mean | 0.8244 | 0.267 | 0.245 |       | 0.242 | 0.522 | 

CV: 0.8244

LB: [0.15, 0.20, 0.20, 0.20, 0.4] -> 0.752
LB: [0.20, 0.20, 0.20, 0.20, 0.4] -> 0.752
LB: [0.25, 0.20, 0.20, 0.20, 0.4] -> 0.752

LB: [0.20, 0.15, 0.20, 0.20, 0.4] -> 0.738

LB: [0.20, 0.25, 0.20, 0.20, 0.4] -> 0.758

LB: [0.20, 0.20, 0.15, 0.20, 0.4] -> 0.751
LB: [0.20, 0.20, 0.25, 0.20, 0.4] -> 0.752

LB: [0.20, 0.20, 0.20, 0.15, 0.4] -> 0.740
LB: [0.20, 0.20, 0.20, 0.25, 0.4] -> 0.758


LB: [0.20, 0.20, 0.20, 0.20, 0.35] -> 0.752

LB: [0.20, 0.30, 0.20, 0.20, 0.4] -> 0.755
LB: [0.20, 0.20, 0.20, 0.30, 0.4] -> 0.753
LB: [0.20, 0.20, 0.20, 0.20, 0.5] -> 0.751

python trace_od.py /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_0_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_0_2862-score-0.8466-at-0.234-0.325-0.278-0.234-0.759.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_1_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_1_1590-score-0.8177-at-0.278-0.278-0.234-0.234-0.325.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_2_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_2_1060-score-0.7953-at-0.430-0.234-0.278-0.234-0.489.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_3_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_3_2438-score-0.8234-at-0.158-0.194-0.278-0.278-0.158.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_4_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_4_2226-score-0.8391-at-0.234-0.194-0.194-0.234-0.278.ckpt 


# DynUnet

| Fold | Score  | AFRT  | BGT   | RBSM  | TRGLB  | VLP    |
|------|--------|-------|-------|-------|--------|--------|
| 0    | 0.8354 | 0.278 | 0.278 | 0.126 | 0.278  | 0.194  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_0_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_0_1360-score-0.8354-at-0.278-0.278-0.126-0.278-0.194.ckpt 
| 1    | 0.8050 | 0.278 | 0.126 | 0.126 | 0.234  | 0.430  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_1_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_1_1360-score-0.8050-at-0.278-0.126-0.126-0.234-0.430.ckpt 
| 2    | 0.7980 | 0.278 | 0.126 | 0.158 | 0.126  | 0.278  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_2_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_2_2560-score-0.7980-at-0.278-0.126-0.158-0.126-0.278.ckpt 
| 3    | 0.8420 | 0.430 | 0.278 | 0.126 | 0.234  | 0.072  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_3_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_3_880-score-0.8420-at-0.430-0.278-0.126-0.234-0.072.ckpt 
| 4    | 0.8324 | 0.194 | 0.158 | 0.072 | 0.194  | 0.097  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_4_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_4_880-score-0.8324-at-0.194-0.158-0.072-0.194-0.097.ckpt 
| Mean | 0,82   | 0,29  | 0,19  | 0,12  | 0,21   | 0,21   | 

python trace_od_dynunet.py /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_0_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_0_1360-score-0.8354-at-0.278-0.278-0.126-0.278-0.194.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_1_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_1_1360-score-0.8050-at-0.278-0.126-0.126-0.234-0.430.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_2_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_2_2560-score-0.7980-at-0.278-0.126-0.158-0.126-0.278.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_3_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_3_880-score-0.8420-at-0.430-0.278-0.126-0.234-0.072.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_4_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_4_880-score-0.8324-at-0.194-0.158-0.072-0.194-0.097.ckpt 

CV: 0.82 +- 0.019

LB [0.20, 0.20, 0.20, 0.20, 0.35] -> 0.741

## DynUnet (Stride 4 & 2)

| Fold | Score  | AFRT  | BGT   | RBSM  | TRGLB | VLP   |
|------|--------|-------|-------|-------|-------|-------|
| 0    | 0.8496 | 0.234 | 0.430 | 0.194 | 0.234 | 0.616 |
| 1    | 0.8079 | 0.325 | 0.234 | 0.376 | 0.278 | 0.551 |
| 2    | 0.7689 | 0.430 | 0.278 | 0.194 | 0.278 | 0.489 |
| 3    | 0.8307 | 0.430 | 0.278 | 0.126 | 0.158 | 0.234 |
| 4    | 0.8263 | 0.325 | 0.278 | 0.126 | 0.194 | 0.126 |

CV 0.81668

0.8496-at-0.234-0.430-0.194-0.234-0.616.ckpt
0.8079-at-0.325-0.234-0.376-0.278-0.551.ckpt
0.7689-at-0.430-0.278-0.194-0.278-0.489.ckpt
0.8307-at-0.430-0.278-0.126-0.158-0.234.ckpt
0.8263-at-0.325-0.278-0.126-0.194-0.126.ckpt

/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_0_rc_ic_denoised/lightning_logs/version_2/checkpoints/dynunet_128_fold_0_1166-score-0.8496-at-0.234-0.430-0.194-0.234-0.616.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_1_rc_ic_denoised/lightning_logs/version_1/checkpoints/dynunet_128_fold_1_1060-score-0.8079-at-0.325-0.234-0.376-0.278-0.551.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_2_rc_ic_denoised/lightning_logs/version_1/checkpoints/dynunet_128_fold_2_636-score-0.7689-at-0.430-0.278-0.194-0.278-0.489.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_3_rc_ic_denoised/lightning_logs/version_1/checkpoints/dynunet_128_fold_3_1696-score-0.8307-at-0.430-0.278-0.126-0.158-0.234.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_4_rc_ic_denoised/lightning_logs/version_1/checkpoints/dynunet_128_fold_4_848-score-0.8263-at-0.325-0.278-0.126-0.194-0.126.ckpt

# Ideas

## Data

- External data
- Flips augmentations - Helps
- Slight rotations along Y & X - Helps
- Noise augmentations
- Copy-paste
- Random erase - Much lower score (?) 0.85 -> 0.75 on fold 0 (maybe bug?)

## Models

- Bigger models
- Stride 4 only - It seems that it is worse than using strides 4 & 2
- Stride 2 only - It seems that it is worse than using strides 4 & 2
- DynUnet
- Stride 4 for Ribosome & Virus-like particle, Stride 2 for the rest
- (--apply_loss_on_each_stride) Independent detection losses on stride 4 and stride 2 (To ensure all channels are used on both strides)

## Training

- Knowledge distillation
- LR & WD Tuning


## Postprocessing

- Winner takes all (current) vs multiple classes per anchor