# Baseline SegResNet

| Fold | Score  | Threshold  |
|------|--------|------------|
| 0    | 0.7810 | 0.0625     |
| 1    | 0.7634 | 0.0625     |
| 2    | 0.7191 | 0.0625     |
| 3    | 0.7980 | 0.0900     |
| 4    | 0.8135 | 0.0900     |

CV: 0.775
LB: 0.736 (Score 0.1. TopK 2048 per class)

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

LB: [0.20. 0.20. 0.20. 0.20. 0.5] -> 0.739
LB: [0.20. 0.20. 0.20. 0.20. 0.6] -> 0.727
LB: [0.20. 0.20. 0.20. 0.20. 0.7] -> 0.710
LB: [0.15. 0.20. 0.20. 0.20. 0.5] -> 0.739
LB: [0.25. 0.20. 0.20. 0.20. 0.5] -> 0.738
LB: [0.20. 0.20. 0.20. 0.20. 0.4] -> 0.745

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

LB: [0.20. 0.20. 0.20. 0.20. 0.4] -> 0.752 BASELINE SUB

LB: [0.15. 0.20. 0.20. 0.20. 0.4] -> 0.752
LB: [0.20. 0.20. 0.20. 0.20. 0.4] -> 0.752
LB: [0.25. 0.20. 0.20. 0.20. 0.4] -> 0.752

LB: [0.20. 0.15. 0.20. 0.20. 0.4] -> 0.738
LB: [0.20. 0.25. 0.20. 0.20. 0.4] -> 0.758
LB: [0.20. 0.30. 0.20. 0.20. 0.4] -> 0.755

LB: [0.20. 0.20. 0.15. 0.20. 0.4] -> 0.751
LB: [0.20. 0.20. 0.25. 0.20. 0.4] -> 0.752

LB: [0.20. 0.20. 0.20. 0.15. 0.4] -> 0.740
LB: [0.20. 0.20. 0.20. 0.25. 0.4] -> 0.758
LB: [0.20. 0.20. 0.20. 0.30. 0.4] -> 0.753

LB: [0.20. 0.20. 0.20. 0.20. 0.35] -> 0.752
LB: [0.20. 0.20. 0.20. 0.20. 0.40] -> 0.752
LB: [0.20. 0.20. 0.20. 0.20. 0.50] -> 0.751

LB: [0.20. 0.25. 0.20. 0.25. 0.40] -> 0.763
LB: [0.20. 0.25. 0.20. 0.25. 0.25] -> 0.764
LB: [0.30. 0.25. 0.20. 0.25. 0.4] -> 0.762
LB: [0.20. 0.20. 0.20. 0.20. 0.3] -> 0.754
LB: [0.15. 0.20. 0.20. 0.20. 0.4] -> 0.752

python trace_od.py /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_0_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_0_2862-score-0.8466-at-0.234-0.325-0.278-0.234-0.759.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_1_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_1_1590-score-0.8177-at-0.278-0.278-0.234-0.234-0.325.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_2_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_2_1060-score-0.7953-at-0.430-0.234-0.278-0.234-0.489.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_3_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_3_2438-score-0.8234-at-0.158-0.194-0.278-0.278-0.158.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_4_rc_ic_denoised/lightning_logs/version_1/checkpoints/segresnetv2_96_fold_4_2226-score-0.8391-at-0.234-0.194-0.194-0.234-0.278.ckpt 


# DynUnet

| Fold | Score  | AFRT  | BGT   | RBSM  | TRGLB  | VLP    |
|------|--------|-------|-------|-------|--------|--------|
| 0    | 0.8354 | 0.278 | 0.278 | 0.126 | 0.278  | 0.194  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_0_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_0_1360-score-0.8354-at-0.278-0.278-0.126-0.278-0.194.ckpt 
| 1    | 0.8050 | 0.278 | 0.126 | 0.126 | 0.234  | 0.430  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_1_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_1_1360-score-0.8050-at-0.278-0.126-0.126-0.234-0.430.ckpt 
| 2    | 0.7980 | 0.278 | 0.126 | 0.158 | 0.126  | 0.278  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_2_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_2_2560-score-0.7980-at-0.278-0.126-0.158-0.126-0.278.ckpt 
| 3    | 0.8420 | 0.430 | 0.278 | 0.126 | 0.234  | 0.072  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_3_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_3_880-score-0.8420-at-0.430-0.278-0.126-0.234-0.072.ckpt 
| 4    | 0.8324 | 0.194 | 0.158 | 0.072 | 0.194  | 0.097  | /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_4_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_4_880-score-0.8324-at-0.194-0.158-0.072-0.194-0.097.ckpt 
| Mean | 0.82   | 0.29  | 0.19  | 0.12  | 0.21   | 0.21   | 

python trace_od_dynunet.py /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_0_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_0_1360-score-0.8354-at-0.278-0.278-0.126-0.278-0.194.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_1_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_1_1360-score-0.8050-at-0.278-0.126-0.126-0.234-0.430.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_2_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_2_2560-score-0.7980-at-0.278-0.126-0.158-0.126-0.278.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_3_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_3_880-score-0.8420-at-0.430-0.278-0.126-0.234-0.072.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_4_rc_ic_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_4_880-score-0.8324-at-0.194-0.158-0.072-0.194-0.097.ckpt 

CV: 0.82 +- 0.019

LB [0.20. 0.20. 0.20. 0.20. 0.35] -> 0.741

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


## DynUnet (Stride 2)

python trace_od_dynunet.py /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_0_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_0_1590-score-0.8345-at-0.234-0.325-0.158-0.278-0.278.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_1_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_1_2332-score-0.8102-at-0.278-0.126-0.194-0.158-0.376.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_2_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_2_742-score-0.7732-at-0.325-0.234-0.158-0.278-0.194.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_3_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_3_1378-score-0.8149-at-0.325-0.325-0.278-0.234-0.158.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_4_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/dynunet_128_fold_4_2650-score-0.8236-at-0.194-0.126-0.126-0.194-0.097.ckpt

| Fold | Score    | AFRT     | BGT      | RBSM     | TRGLB    | VLP      |
|------|----------|----------|----------|----------|----------|----------|
| 0    | 0.8345   | 0.234    | 0.325    | 0.158    | 0.278    | 0.278    |  
| 1    | 0.8102   | 0.278    | 0.126    | 0.194    | 0.158    | 0.376    |  
| 2    | 0.7732   | 0.325    | 0.234    | 0.158    | 0.278    | 0.194    |  
| 3    | 0.8149   | 0.325    | 0.325    | 0.278    | 0.234    | 0.158    |
| 4    | 0.8236   | 0.194    | 0.126    | 0.126    | 0.194    | 0.097    |
|------|----------|----------|----------|----------|----------|----------|
| MEAN | 0.811280 | 0.271200 | 0.227200 | 0.182800 | 0.228400 | 0.220600 |

## SegResNet (Stride 2)

python trace_od.py /home/ekhvedchenia/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_0_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_0_1562-score-0.8465-at-0.194-0.376-0.194-0.194-0.325.ckpt /home/ekhvedchenia/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_3_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_3_1846-score-0.8298-at-0.158-0.325-0.278-0.278-0.278.ckpt /home/ekhvedchenia/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_1_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_1_710-score-0.8239-at-0.376-0.430-0.234-0.234-0.278.ckpt /home/ekhvedchenia/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_2_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_2_1349-score-0.8109-at-0.278-0.234-0.234-0.325-0.234.ckpt /home/ekhvedchenia/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_4_rc_ic_s2_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96_fold_4_1278-score-0.8137-at-0.158-0.234-0.194-0.194-0.376.ckpt

| Fold | Score  | AFRT  | BGT   | RBSM       | TRGLB      | VLP        |
|------|--------|-------|-------|------------|------------|------------|
| 0    | 0.8465 | 0.194 | 0.376 | 0.194      | 0.194      | 0.325      |
| 3    | 0.8298 | 0.158 | 0.325 | 0.278      | 0.278      | 0.278      |
| 1    | 0.8239 | 0.376 | 0.430 | 0.234      | 0.234      | 0.278      |
| 2    | 0.8109 | 0.278 | 0.234 | 0.234      | 0.325      | 0.234      |
| 4    | 0.8137 | 0.158 | 0.234 | 0.194      | 0.194      | 0.376      |
|------| ------ | ------| ----- | ---------- | ---------- | ---------- |
| AVG  | 0.8249 | 0.232 | 0.319 | 0.226      | 0.2450     | 0.298      |


Mean of two 
0.818120 0.252000 0.273500 0.204800 0.236700 0.259400

LB: [0.252. 0.273. 0.204. 0.236. 0.259] -> 0.774
LB: [0.200. 0.273. 0.204. 0.236. 0.259] -> 0.774 (higher)
LB: [0.252. 0.250. 0.204. 0.236. 0.259] -> 0.774 (lower)
LB: [0.252. 0.273. 0.250. 0.236. 0.259] -> 0.772
LB: [0.252. 0.273. 0.204. 0.200. 0.259] -> 0.772
LB: [0.252. 0.273. 0.204. 0.300. 0.259] -> 0.770
LB: [0.252. 0.300. 0.204. 0.236. 0.259] -> 0.773
LB: [0.252. 0.273. 0.204. 0.236. 0.200] -> 0.773
LB: [0.252. 0.273. 0.204. 0.236. 0.300] -> 0.772

## od_segresnet_s1_fold_0_rc_ic_denoised Ablation

| Experiment                                  | Fold | Score  |
|---------------------------------------------|------|--------|
| Baseline                                    | 0    | 0.8523 |
| with EMA                                    | 0    | 0.8413 |
| --random_erase_prob=0.25                    | 0    | 0.8326 |
| --copy_paste_prob=0.25                      | 0    | 0.8343 |
| --y_rotation_limit=20 --x_rotation_limit=20 | 0    | 0.8447 |
| --use_single_label_per_anchor=False         | 0    | 0.8388 |
| --use_centernet_nms=False                   | 0    | 0.841  |

## od_segresnet_fold_0_rc_ic_denoised Ablation

| Experiment                         | Fold | Score |
|------------------------------------|------|-------|
| --use_stride4=False                | 0    |       |
| --use_stride2=False                | 0    |       |
| All strides (baseline)             | 0    |       |
| All strides with loss per stride   | 0    |       |
| --copy_paste_prob=0.25 (Only rare) | 0    |       |


## 13 Jan 2025

- Validation uses same tiling scheme as inference kernel
- 10.012 instead of 10.0

### V3 SegResNet S2 

python trace_od_segresnet.py /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_0/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_0_1280-score-0.8429-at-0.325-0.278-0.194-0.278-0.551.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_1/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_1_1440-score-0.8215-at-0.126-0.194-0.194-0.234-0.194.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_2/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_2_4320-score-0.7893-at-0.234-0.158-0.278-0.194-0.325.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_3/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_3_2560-score-0.8413-at-0.278-0.551-0.376-0.158-0.278.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_4/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_4_3680-score-0.8426-at-0.278-0.194-0.158-0.126-0.158.ckpt


/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_0/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_0_1280-score-0.8429-at-0.325-0.278-0.194-0.278-0.551.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_1/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_1_1440-score-0.8215-at-0.126-0.194-0.194-0.234-0.194.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_2/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_2_4320-score-0.7893-at-0.234-0.158-0.278-0.194-0.325.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_3/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_3_2560-score-0.8413-at-0.278-0.551-0.376-0.158-0.278.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_segresnetv2_fold_4/adamw_torch_1e-04_0.01/_rc_ic_s2_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/segresnetv2_96x128x128_fold_4_3680-score-0.8426-at-0.278-0.194-0.158-0.126-0.158.ckpt

| Fold | Score    | AFRT    | BGT      | RBSM     | TRGLB     | VLP     |
|------|----------|---------|----------|----------|-----------|---------|
| 0    | 0.8429   | 0.325   | 0.278    | 0.194    | 0.278     | 0.551   |
| 1    | 0.8215   | 0.126   | 0.194    | 0.194    | 0.234     | 0.194   |
| 2    | 0.7893   | 0.234   | 0.158    | 0.278    | 0.194     | 0.325   |
| 3    | 0.8413   | 0.278   | 0.551    | 0.376    | 0.158     | 0.278   |
| 4    | 0.8426   | 0.278   | 0.194    | 0.158    | 0.126     | 0.158   |
|------| -------- | ------- | -------- | -------- | --------- | ------- |
| Mean | 0.82752  | 0.2482  | 0.275    | 0.24     | 0.198     | 0.3012  |

### V3 DynUnet S2

python trace_od_dynunet.py /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_0/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_0_1590-score-0.8442-at-0.278-0.325-0.194-0.234-0.551.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_1/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_1_2332-score-0.8098-at-0.325-0.097-0.278-0.278-0.325.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_2/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_2_2120-score-0.8030-at-0.325-0.097-0.126-0.278-0.376.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_3/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_3_1802-score-0.8185-at-0.430-0.234-0.376-0.194-0.194.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_4/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_4_954-score-0.8323-at-0.325-0.325-0.278-0.325-0.325.ckpt

/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_0/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_0_1590-score-0.8442-at-0.278-0.325-0.194-0.234-0.551.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_1/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_1_2332-score-0.8098-at-0.325-0.097-0.278-0.278-0.325.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_2/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_2_2120-score-0.8030-at-0.325-0.097-0.126-0.278-0.376.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_3/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_3_1802-score-0.8185-at-0.430-0.234-0.376-0.194-0.194.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_dynunet_fold_4/adamw_torch_3e-04_0.0001_ema_0.995_10/_rc_ic_s2_slpa_cnms_drop_0.05_denoised/lightning_logs/version_0/checkpoints/dynunet_96x128x128_fold_4_954-score-0.8323-at-0.325-0.325-0.278-0.325-0.325.ckpt

| Fold | Score    | AFRT    | BGT     | RBSM    | TRGLB  | VLP    |
|------|----------|---------|---------|---------|--------|--------|
| 0    | 0.8442   | 0.278   | 0.325   | 0.194   | 0.234  | 0.551  |
| 1    | 0.8098   | 0.325   | 0.097   | 0.278   | 0.278  | 0.325  |
| 2    | 0.8030   | 0.325   | 0.097   | 0.126   | 0.278  | 0.376  |
| 3    | 0.8185   | 0.430   | 0.234   | 0.376   | 0.194  | 0.194  |
| 4    | 0.8323   | 0.325   | 0.325   | 0.278   | 0.325  | 0.325  |
|------|----------|---------|---------|---------|--------|--------|
| MEAN | 0.82156  | 0.3366  | 0.2156  | 0.2504  | 0.2618 | 0.3542 |

Mean of two
0.82454 0.2924	0.2453	0.2452	0.2299	0.3277

### V3 HRNetW18 S2

python trace_od_hrnet.py /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_1/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_1_4664-score-0.7916-at-0.234-0.158-0.278-0.234-0.325.ckpt  /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_3/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_3_7738-score-0.8223-at-0.194-0.234-0.234-0.194-0.158.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_2/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_2_2650-score-0.7981-at-0.234-0.194-0.325-0.194-0.194.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_0/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_0_6254-score-0.8406-at-0.325-0.126-0.194-0.158-0.278.ckpt /home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_4/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_4_6678-score-0.8254-at-0.234-0.158-0.194-0.194-0.097.ckpt 

/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_1/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_1_4664-score-0.7916-at-0.234-0.158-0.278-0.234-0.325.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_3/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_3_7738-score-0.8223-at-0.194-0.234-0.234-0.194-0.158.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_2/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_2_2650-score-0.7981-at-0.234-0.194-0.325-0.194-0.194.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_0/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_0_6254-score-0.8406-at-0.325-0.126-0.194-0.158-0.278.ckpt
/home/bloodaxe/develop/Kaggle-2024-CryoET/runs/od_hrnet_fold_4/adamw_torch_5e-05_0.0001/_rc_ic_s2_s4_slpa_cnms_denoised/lightning_logs/version_0/checkpoints/hrnet_96x128x128_fold_4_6678-score-0.8254-at-0.234-0.158-0.194-0.194-0.097.ckpt


| Fold | Score  | AFRT   | BGT     | RBSM    | TRGLB   | VLP     |
|------|--------|--------|---------|---------|---------|---------|
| 0    | 0.8406 | 0.325  | 0.126   | 0.194   | 0.158   | 0.278   |
| 1    | 0.7916 | 0.234  | 0.158   | 0.278   | 0.234   | 0.325   |
| 2    | 0.7981 | 0.234  | 0.194   | 0.325   | 0.194   | 0.194   |
| 3    | 0.8223 | 0.194  | 0.234   | 0.234   | 0.194   | 0.158   |
| 4    | 0.8254 | 0.234  | 0.158   | 0.194   | 0.194   | 0.097   |
|------|--------|--------|---------|---------|---------|---------|
| MEAN | 0.8156 | 0.2442 | 0.174   | 0.245   | 0.1948  | 0.2104  |

3 Models average 

0,82156	0,2763333333	0,2215333333	0,2451333333	0,2182	0,2886

V3 SegResNet and DynUnet S2

#1 [0.2924, 0.2453, 0.2452, 0.2299, 0.3277] - 0.776
#2 [0.2724, 0.2453, 0.2452, 0.2299, 0.3277] - 0.776 (Higher than #1)
#3 [0.2924, 0.2253, 0.2452, 0.2299, 0.3277] - 0.775
#4 [0.2924, 0.2453, 0.2252, 0.2299, 0.3277] - 0.776 (Higher than #5) best
#5 [0.2924, 0.2453, 0.2452, 0.2099, 0.3277] - 0.776 (Higher than #2)

#6 [0.2524, 0.2453, 0.2452, 0.2299, 0.3277] # Best overall
#7 [0.2724, 0.2653, 0.2452, 0.2299, 0.3277] # Worse
#8 [0.2924, 0.2453, 0.2052, 0.2299, 0.3277] Second best
#9 [0.2924, 0.2453, 0.2452, 0.1899, 0.3277] 0.772
#10 [0.2924, 0.2453, 0.2452, 0.2299, 0.3077] Third best

#11 [0.2524, 0.2453, 0.2052, 0.2099, 0.3077] 0.775
#12 [0.2524, 0.2453, 0.2052, 0.2099, 0.3077] 0.746 use_weighted_average=True (Implementation bug?)
#13 [0.2524, 0.2453, 0.2052, 0.2099, 0.3077] 0.778 window_step=(192, 79, 79) instead window_step=(192, 90, 90)
#14 [0.2524, 0.2453, 0.2052, 0.2099, 0.3077] 0.775 centernet_heatmap_nms kernel 5,3,3
#15 [0.2524, 0.2453, 0.2052, 0.2099, 0.3077] 0.707 normalize_volume_to_percentile_range



## V4 SegResNet S2 on 6 classes

```
python summarize_checkpoints.py  runs/segresnetv2_fold_0_6x96x128x128_rc_ic_s2/250117_2136_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_0_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_2136_segresnetv2_fold_0_6x96x128x128_rc_ic_s2_2560-score-0.8457-at-0.265-0.290-0.195-0.150-0.550.ckpt  runs/segresnetv2_fold_1_6x96x128x128_rc_ic_s2/250117_2225_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_1_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_2225_segresnetv2_fold_1_6x96x128x128_rc_ic_s2_2880-score-0.8366-at-0.345-0.110-0.185-0.135-0.305.ckpt  runs/segresnetv2_fold_2_6x96x128x128_rc_ic_s2/250117_1623_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_2_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_1623_segresnetv2_fold_2_6x96x128x128_rc_ic_s2_2240-score-0.8046-at-0.355-0.280-0.395-0.340-0.185.ckpt  runs/segresnetv2_fold_3_6x96x128x128_rc_ic_s2/250118_0038_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_3_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250118_0038_segresnetv2_fold_3_6x96x128x128_rc_ic_s2_2240-score-0.8398-at-0.230-0.345-0.405-0.360-0.255.ckpt  runs/segresnetv2_fold_4_6x96x128x128_rc_ic_s2/250118_0108_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_4_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250118_0108_segresnetv2_fold_4_6x96x128x128_rc_ic_s2_2880-score-0.8437-at-0.165-0.350-0.245-0.235-0.270.ckpt
python trace_od_segresnet.py --num_classes=6 runs/segresnetv2_fold_0_6x96x128x128_rc_ic_s2/250117_2136_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_0_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_2136_segresnetv2_fold_0_6x96x128x128_rc_ic_s2_2560-score-0.8457-at-0.265-0.290-0.195-0.150-0.550.ckpt  runs/segresnetv2_fold_1_6x96x128x128_rc_ic_s2/250117_2225_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_1_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_2225_segresnetv2_fold_1_6x96x128x128_rc_ic_s2_2880-score-0.8366-at-0.345-0.110-0.185-0.135-0.305.ckpt  runs/segresnetv2_fold_2_6x96x128x128_rc_ic_s2/250117_1623_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_2_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_1623_segresnetv2_fold_2_6x96x128x128_rc_ic_s2_2240-score-0.8046-at-0.355-0.280-0.395-0.340-0.185.ckpt  runs/segresnetv2_fold_3_6x96x128x128_rc_ic_s2/250118_0038_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_3_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250118_0038_segresnetv2_fold_3_6x96x128x128_rc_ic_s2_2240-score-0.8398-at-0.230-0.345-0.405-0.360-0.255.ckpt  runs/segresnetv2_fold_4_6x96x128x128_rc_ic_s2/250118_0108_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_4_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250118_0108_segresnetv2_fold_4_6x96x128x128_rc_ic_s2_2880-score-0.8437-at-0.165-0.350-0.245-0.235-0.270.ckpt
```

| fold         |    score |   AFRT |   BGT |   RBSM |   TRGLB |   VLP | checkpoint                                                                                              |
|:-------------|---------:|-------:|------:|-------:|--------:|------:|:--------------------------------------------------------------------------------------------------------|
| 0            | 0.8457   |  0.265 | 0.29  |  0.195 |   0.15  | 0.55  | 250117_2136_segresnetv2_fold_0_6x96x128x128_rc_ic_s2_2560-score-0.8457-at-0.265-0.290-0.195-0.150-0.550 |
| 1            | 0.8366   |  0.345 | 0.11  |  0.185 |   0.135 | 0.305 | 250117_2225_segresnetv2_fold_1_6x96x128x128_rc_ic_s2_2880-score-0.8366-at-0.345-0.110-0.185-0.135-0.305 |
| 2            | 0.8046   |  0.355 | 0.28  |  0.395 |   0.34  | 0.185 | 250117_1623_segresnetv2_fold_2_6x96x128x128_rc_ic_s2_2240-score-0.8046-at-0.355-0.280-0.395-0.340-0.185 |
| 3            | 0.8398   |  0.23  | 0.345 |  0.405 |   0.36  | 0.255 | 250118_0038_segresnetv2_fold_3_6x96x128x128_rc_ic_s2_2240-score-0.8398-at-0.230-0.345-0.405-0.360-0.255 |
| 4            | 0.8437   |  0.165 | 0.35  |  0.245 |   0.235 | 0.27  | 250118_0108_segresnetv2_fold_4_6x96x128x128_rc_ic_s2_2880-score-0.8437-at-0.165-0.350-0.245-0.235-0.270 |
| mean         | 0.83408  |  0.272 | 0.275 |  0.285 |   0.244 | 0.313 |                                                                                                         |
| mean (curve) | 0.831443 |  0.265 | 0.28  |  0.245 |   0.215 | 0.36  | 

## V4 DynUnet S2 on 6 classes

```
python summarize_checkpoints.py  runs/dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05/250118_0912_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_0912_dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05_1484-score-0.8272-at-0.145-0.330-0.215-0.235-0.405.ckpt  runs/dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05/250118_0953_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_0953_dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05_4982-score-0.8337-at-0.390-0.185-0.215-0.210-0.235.ckpt  runs/dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05/250118_1112_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1112_dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05_2650-score-0.7971-at-0.425-0.160-0.550-0.275-0.650.ckpt  runs/dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05/250118_1208_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1208_dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05_1590-score-0.8418-at-0.385-0.345-0.345-0.305-0.210.ckpt  runs/dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05/250118_1253_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1253_dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05_1060-score-0.8420-at-0.230-0.340-0.255-0.375-0.140.ckpt
python trace_od_dynunet_s2.py --num_classes=6 runs/dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05/250118_0912_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_0912_dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05_1484-score-0.8272-at-0.145-0.330-0.215-0.235-0.405.ckpt  runs/dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05/250118_0953_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_0953_dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05_4982-score-0.8337-at-0.390-0.185-0.215-0.210-0.235.ckpt  runs/dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05/250118_1112_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1112_dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05_2650-score-0.7971-at-0.425-0.160-0.550-0.275-0.650.ckpt  runs/dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05/250118_1208_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1208_dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05_1590-score-0.8418-at-0.385-0.345-0.345-0.305-0.210.ckpt  runs/dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05/250118_1253_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1253_dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05_1060-score-0.8420-at-0.230-0.340-0.255-0.375-0.140.ckpt
```

| fold         |    score |   AFRT |   BGT |   RBSM |   TRGLB |   VLP | checkpoint                                                                                                  |
|:-------------|---------:|-------:|------:|-------:|--------:|------:|:------------------------------------------------------------------------------------------------------------|
| 0            | 0.8272   |  0.145 | 0.33  |  0.215 |   0.235 | 0.405 | 250118_0912_dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05_1484-score-0.8272-at-0.145-0.330-0.215-0.235-0.405 |
| 1            | 0.8337   |  0.39  | 0.185 |  0.215 |   0.21  | 0.235 | 250118_0953_dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05_4982-score-0.8337-at-0.390-0.185-0.215-0.210-0.235 |
| 2            | 0.7971   |  0.425 | 0.16  |  0.55  |   0.275 | 0.65  | 250118_1112_dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05_2650-score-0.7971-at-0.425-0.160-0.550-0.275-0.650 |
| 3            | 0.8418   |  0.385 | 0.345 |  0.345 |   0.305 | 0.21  | 250118_1208_dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05_1590-score-0.8418-at-0.385-0.345-0.345-0.305-0.210 |
| 4            | 0.842    |  0.23  | 0.34  |  0.255 |   0.375 | 0.14  | 250118_1253_dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05_1060-score-0.8420-at-0.230-0.340-0.255-0.375-0.140 |
| mean         | 0.82836  |  0.315 | 0.272 |  0.316 |   0.28  | 0.328 |                                                                                                             |
| mean (curve) | 0.825544 |  0.235 | 0.33  |  0.22  |   0.26  | 0.47  |  


```
python summarize_checkpoints.py  runs/dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05/250118_0912_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_0912_dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05_1484-score-0.8272-at-0.145-0.330-0.215-0.235-0.405.ckpt  runs/dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05/250118_0953_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_0953_dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05_4982-score-0.8337-at-0.390-0.185-0.215-0.210-0.235.ckpt  runs/dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05/250118_1112_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1112_dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05_2650-score-0.7971-at-0.425-0.160-0.550-0.275-0.650.ckpt  runs/dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05/250118_1208_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1208_dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05_1590-score-0.8418-at-0.385-0.345-0.345-0.305-0.210.ckpt  runs/dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05/250118_1253_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1253_dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05_1060-score-0.8420-at-0.230-0.340-0.255-0.375-0.140.ckpt runs/segresnetv2_fold_0_6x96x128x128_rc_ic_s2/250117_2136_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_0_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_2136_segresnetv2_fold_0_6x96x128x128_rc_ic_s2_2560-score-0.8457-at-0.265-0.290-0.195-0.150-0.550.ckpt  runs/segresnetv2_fold_1_6x96x128x128_rc_ic_s2/250117_2225_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_1_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_2225_segresnetv2_fold_1_6x96x128x128_rc_ic_s2_2880-score-0.8366-at-0.345-0.110-0.185-0.135-0.305.ckpt  runs/segresnetv2_fold_2_6x96x128x128_rc_ic_s2/250117_1623_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_2_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_1623_segresnetv2_fold_2_6x96x128x128_rc_ic_s2_2240-score-0.8046-at-0.355-0.280-0.395-0.340-0.185.ckpt  runs/segresnetv2_fold_3_6x96x128x128_rc_ic_s2/250118_0038_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_3_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250118_0038_segresnetv2_fold_3_6x96x128x128_rc_ic_s2_2240-score-0.8398-at-0.230-0.345-0.405-0.360-0.255.ckpt  runs/segresnetv2_fold_4_6x96x128x128_rc_ic_s2/250118_0108_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_4_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250118_0108_segresnetv2_fold_4_6x96x128x128_rc_ic_s2_2880-score-0.8437-at-0.165-0.350-0.245-0.235-0.270.ckpt
```

Computed thresholds

| fold         |    score |    AFRT |    BGT |   RBSM |   TRGLB |    VLP |
|:-------------|---------:|--------:|-------:|-------:|--------:|-------:|
| mean         | 0.83122  |  0.2935 | 0.2735 | 0.3005 |   0.262 | 0.3205 |
| mean (curve) | 0.826882 |    0.23 |   0.28 | 0.22   |   0.23  |   0.39 |

# Ideas

## Data

- External data
- Flips augmentations - Helps
- Slight rotations along Y & X - Helps
- Noise augmentations
- Copy-paste
- Random erase - Much lower score (?) 0.85 -> 0.75 on fold 0 (maybe bug?)
- Pretrain on other modalities and then fine-tune on 'denoised' (--transfer_weights)

## Models

- Bigger models
- Stride 4 only - It seems that it is worse than using strides 4 & 2
- Stride 2 only - It seems that it is worse than using strides 4 & 2
- DynUnet
- Stride 4 for Ribosome & Virus-like particle. Stride 2 for the rest
- (--apply_loss_on_each_stride) Independent detection losses on stride 4 and stride 2 (To ensure all channels are used on both strides)

## Training

- Knowledge distillation
- LR & WD Tuning


## Postprocessing

- Individual thresholds per class (Helps!!)
- (--use_single_label_per_anchor) Winner takes all (current) vs multiple classes per anchor. --use_single_label_per_anchor=False does not seems to improve score