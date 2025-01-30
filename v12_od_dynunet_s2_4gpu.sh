for FOLD in 0 1 2 3 4 5 6
do
  torchrun --standalone --nproc-per-node=4 train_od.py \
  --anisotropic_scale_limit=0.07 \
  --average_tokens_across_devices=True \
  --dataloader_num_workers=8 \
  --dataloader_persistent_workers=True \
  --dataloader_pin_memory=True \
  --ddp_find_unused_parameters=True \
  --early_stopping=25 \
  --fold=$FOLD \
  --learning_rate=3e-4 \
  --max_grad_norm=3 \
  --ddp_find_unused_parameters=True \
  --model_name=dynunet  \
  --num_train_epochs=75 \
  --per_device_train_batch_size=6 \
  --random_erase_prob=0.2 \
  --copy_paste_prob=0.25 \
  --copy_paste_limit=2 \
  --mixup_prob=0.25 \
  --scale_limit=0.15 \
  --use_6_classes=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --version_prefix=v12_ \
  --assigned_min_iou_for_anchor=0.5 \
  --warmup_steps=128 \
  --weight_decay=0.0001 \
  --x_rotation_limit=45 \
  --x_rotation_limit=45 \
  --seed=$FOLD \
  --interpolation_mode=1 \
  --validate_on_rot90=True
done