for FOLD in 0 1 2 3 4
do
  torchrun --standalone --nproc-per-node=4 train_od.py \
  --anisotropic_scale_limit=0.02 \
  --average_tokens_across_devices=True \
  --dataloader_num_workers=4 \
  --dataloader_persistent_workers=True \
  --dataloader_pin_memory=True \
  --ddp_find_unused_parameters=True \
  --early_stopping=25 \
  --fold=$FOLD \
  --learning_rate=3e-4 \
  --max_grad_norm=3 \
  --model_name=dynunet  \
  --num_train_epochs=75 \
  --per_device_train_batch_size=4 \
  --random_erase_prob=0.2 \
  --scale_limit=0.05 \
  --use_6_classes=True \
  --use_cross_entropy_loss=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --assigned_min_iou_for_anchor=0.9 \
  --version_prefix=v10_ \
  --warmup_steps=256 \
  --weight_decay=0.001 \
  --x_rotation_limit=5 \
  --x_rotation_limit=5 \
  --seed=4444 \
  --validate_on_x_flips=True \
  --validate_on_y_flips=True \
  --validate_on_z_flips=True \
  --bf16=False \
  --fp16=True

done