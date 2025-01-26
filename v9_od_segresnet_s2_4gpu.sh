# Fold 0 already trained
for FOLD in 0 1 2 3 4
do
  torchrun --standalone --nproc-per-node=4 train_od.py \
  --adam_beta1=0.95 --adam_beta2=0.99 \
  --anisotropic_scale_limit=0.02 \
  --average_tokens_across_devices=True \
  --dataloader_num_workers=4 \
  --dataloader_persistent_workers=True \
  --dataloader_pin_memory=True \
  --ddp_find_unused_parameters=True \
  --early_stopping=25 \
  --fold=$FOLD \
  --learning_rate=1e-3 \
  --optim=sgd --ema \
  --max_grad_norm=3 \
  --ddp_find_unused_parameters=True \
  --model_name=segresnetv2 \
  --num_train_epochs=75 \
  --per_device_train_batch_size=4 \
  --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
  --random_erase_prob=0.2 \
  --scale_limit=0.1 \
  --use_6_classes=True \
  --use_cross_entropy_loss=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --version_prefix=v9_ \
  --warmup_steps=128 \
  --weight_decay=0.00 \
  --y_rotation_limit=10 \
  --x_rotation_limit=10 \
  --validate_on_rot90=True \
  --mixup_prob=0.2 --copy_paste_prob=0.2
done

