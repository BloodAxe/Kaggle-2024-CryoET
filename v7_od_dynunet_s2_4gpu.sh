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
  --per_device_train_batch_size=6 \
  --random_erase_prob=0.2 \
  --scale_limit=0.05 \
  --use_6_classes=True \
  --use_cross_entropy_loss=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --version_prefix=v7_ \
  --warmup_steps=64 \
  --weight_decay=0.001 \
  --x_rotation_limit=10 \
  --y_rotation_limit=10
done