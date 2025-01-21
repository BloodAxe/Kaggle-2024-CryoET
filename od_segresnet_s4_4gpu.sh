# for each fold
for FOLD in 1 2 3 4
do

  torchrun --standalone --nproc-per-node=4 train_od.py \
    --model_name=segresnetv2 \
    --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-4 --weight_decay=0.01 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --y_rotation_limit=10 --x_rotation_limit=10 \
    --num_train_epochs=75 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
    --fold=$FOLD  --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_stride2=True --use_stride4=False  --use_cross_entropy_loss=True \
    --copy_paste_prob=0.2 --random_erase_prob=0.2 --mixup_prob=0.2  --use_cross_entropy_loss=True \
    --use_6_classes=True

done
