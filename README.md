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

LB: [0.20, 0.20, 0.20, 0.20, 0.4] -> 0.752
LB: [0.15, 0.20, 0.20, 0.20, 0.4] -> 0.752
LB: [0.20, 0.15, 0.20, 0.20, 0.4] -> 0.738
LB: [0.20, 0.20, 0.15, 0.20, 0.4] -> 0.751
LB: [0.20, 0.20, 0.20, 0.15, 0.4] -> 0.740
LB: [0.20, 0.20, 0.20, 0.25, 0.4] -> 0.758
LB: [0.20, 0.20, 0.25, 0.20, 0.4] -> 
LB: [0.20, 0.25, 0.20, 0.20, 0.4] -> 


python trace_od.py /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_0_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_0_2862-score-0.8466-at-0.234-0.325-0.278-0.234-0.759.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_1_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_1_1590-score-0.8177-at-0.278-0.278-0.234-0.234-0.325.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_2_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_2_1060-score-0.7953-at-0.430-0.234-0.278-0.234-0.489.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_3_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_3_2438-score-0.8234-at-0.158-0.194-0.278-0.278-0.158.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_4_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_4_2226-score-0.8391-at-0.234-0.194-0.194-0.234-0.278.ckpt 

# Ideas

## Data

- External data
- Flips augmentations - Helps
- Slight rotations along Y & X - Helps
- Noise augmentations
- Copy-paste & random erase

## Models

- Bigger models

## Training

- Knowledge distillation
- LR & WD Tuning


## Postprocessing

- Winner takes all (current) vs multiple classes per anchor