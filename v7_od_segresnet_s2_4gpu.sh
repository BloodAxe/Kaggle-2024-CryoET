# for each fold
for FOLD in 0 1 2 3 4
do
  for USE_6_CLASSES in True False
  do
    for USE_OFFSET_HEAD in True False
    do

      torchrun --standalone --nproc-per-node=4 train_od.py \
        --model_name=segresnetv2 \
        --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
        --per_device_train_batch_size=4 --learning_rate=1e-4 --weight_decay=0.01 \
        --adam_beta1=0.95 --adam_beta2=0.99 \
        --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
        --y_rotation_limit=10 --x_rotation_limit=10 \
        --num_train_epochs=75 \
        --warmup_steps=64 \
        --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
        --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_stride4=False \
        --copy_paste_prob=0.2 --random_erase_prob=0.2 --mixup_prob=0.2 \
        --anisotropic_scale_limit=0.02 --scale_limit=0.1 \
        --fold=$FOLD \
        --use_offset_head=$USE_OFFSET_HEAD \
        --use_6_classes=$USE_6_CLASSES \
        --version_prefix=v7_

    done
  done
done