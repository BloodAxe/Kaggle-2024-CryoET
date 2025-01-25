for FOLD in 0 1 2 3 4
do
  torchrun --standalone --nproc-per-node=4 train_od.py \
  --anisotropic_scale_limit=0.1 \
  --average_tokens_across_devices=True  \
  --dataloader_num_workers=4 \
  --dataloader_persistent_workers=True \
  --dataloader_pin_memory=True \
  --ddp_find_unused_parameters=True \
  --early_stopping=25 \
  --fold=2 \
  --learning_rate=1e-5 \
  --max_grad_norm=3 \
  --model_name=hrnet  \
  --num_train_epochs=75 \
  --per_device_train_batch_size=8 \
  --random_erase_prob=0.1 \
  --scale_limit=0.25 \
  --use_6_classes=True \
  --use_cross_entropy_loss=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --version_prefix=v7_ \
  --warmup_steps=64 \
  --weight_decay=0.0001 \
  --x_rotation_limit=25 \
  --y_rotation_limit=25 --seed=555
done
