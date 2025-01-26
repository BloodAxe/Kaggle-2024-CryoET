for FOLD in 0 1 2 3 4
do
  torchrun --standalone --nproc-per-node=3 train_od.py \
  --adam_beta1=0.95 --adam_beta2=0.99 \
  --anisotropic_scale_limit=0.02 \
  --average_tokens_across_devices=True \
  --dataloader_num_workers=4 \
  --dataloader_persistent_workers=True \
  --dataloader_pin_memory=True \
  --ddp_find_unused_parameters=True \
  --early_stopping=25 \
  --fold=$FOLD \
  --learning_rate=1e-4 \
  --max_grad_norm=3 \
  --ddp_find_unused_parameters=True \
  --model_name=segresnetv2 \
  --num_train_epochs=75 \
  --per_device_train_batch_size=8 \
  --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
  --random_erase_prob=0.2 \
  --scale_limit=0.1 \
  --use_6_classes=True \
  --use_cross_entropy_loss=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --version_prefix=v9_ \
  --warmup_steps=64 \
  --ema \
  --weight_decay=0.01 \
  --y_rotation_limit=10 \
  --x_rotation_limit=10 \
  --validate_on_x_flips=True \
  --validate_on_y_flips=True \
  --validate_on_z_flips=False \
  --validate_on_rot90=True
done

