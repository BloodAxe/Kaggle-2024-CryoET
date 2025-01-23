for USE_6_CLASSES in True False
do
  for FOLD in 0 1 2 3 4
  do
    torchrun --standalone --nproc-per-node=3 train_od.py \
    --model_name=hrnet  \
    --per_device_train_batch_size=16 \
    --average_tokens_across_devices=True  \
    --dataloader_num_workers=4 \
    --dataloader_persistent_workers=True \
    --dataloader_pin_memory=True \
    --ddp_find_unused_parameters=True \
    --early_stopping=25 \
    --fold=$FOLD \
    --learning_rate=1e-5 \
    --max_grad_norm=3 \
    --mixup_prob=0.1 \
    --random_erase_prob=0.1 \
    --copy_paste_prob=0.1 \
    --num_train_epochs=75 \
    --use_6_classes=$USE_6_CLASSES \
    --use_cross_entropy_loss=True \
    --use_instance_crops=True \
    --use_random_crops=True \
    --use_stride4=False \
    --use_offset_head=False \
    --warmup_steps=64 \
    --x_rotation_limit=10 \
    --y_rotation_limit=10 \
    --anisotropic_scale_limit=0.02 --scale_limit=0.05 \
    --version_prefix=v7_
  done
done