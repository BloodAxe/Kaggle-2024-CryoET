Introduction

My baseline solution was a segmentation model built from the SegResNet backbone. As a supervised objective target, I built a five-class heatmap with Gaussian peaks for present ground-truth targets in volume. To train a model I used reduced focal loss from CenterNet paper and NMS postprocessing approach from the same paper. Although this approach immediately gave ~0.740 LB score, I ditched it as I anticipated many participants would be using the segmentation approach as the easiest one. So if I decide to team up in the future, there will be less diversity in the final model ensemble.

Modeling approach 

My final approach uses SegResNet and DynUnet backbones from Monai in combination with a custom point detection head. Based on my experience creating YOLO-based object detection models, I implemented an anchor-free point detection model. Unlike in the segmentation approach, the loss function is quite complicated and includes:
Assignment of ground-truth labels to predicted centers based on point-point IoU. Point-Point IoU metric term defined as exp(-mse(x,y)/ (2 * radius ^ 2))
Taking Top-K predictions for each GT label for training
Computing varifocal loss for class map prediction and IoU-based loss for distance regression
My point detection approach worked well in terms of accuracy and training speed. On 4x3090 it took a mere two hours to train a single fold. The first submission of a single-fold detection model scored 0.752 on the LB. Once the modeling approach became clear, the next effort was to make it fast.

In this competition, we need to process 500 scans within a 12-hour time limit.  This translates into ~1m30s per single study. So 
In my models, I predict class-map [B,C,D/2,H/2,W/2] and offsets map [B,3,D/2,H/2,W/2] in stride 2 along depth, height, and width. 

For the object detection approach, there is no need to predict a full-resolution feature maps. Predicting smaller feature maps has a massive impact on model throughput. I've got almost 50% speed increase when changing to stride 2 output from stride 1 (full resolution).

I ablate on stride 1, stride 2, stride 4, and stride 2 & 4 outputs and found that:
Stride 1 gives a slightly, slightly better (0.002) LB  score than stride 2
Stride 4 also worked ok, but stride 2 was better in the F-beta score.
Using stride 2 and 4 didn't bring any improvements over using only stride 2. An interesting observation when using the two-heads method: Large particles tend to migrate to stride 4 while small classes were present on the class-map with stride 2.

In all my models I use stride 2 prediction maps.

Training

Tiling

Object Detection Ensemble

My final object ensemble is 5 models (folds) of OD SegResNet and 5 models of OD DynUnet (OD stands for Object Detection to differentiate them from segmentation-based models). A 10 models in total. Model averaging happens at classmap/offsets 

