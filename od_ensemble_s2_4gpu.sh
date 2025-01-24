for FOLD in 0 1 2 3 4
do
  torchrun --standalone --nproc-per-node=4 train_od_ensemble.py \
  --model_name=ensemble  \
  --per_device_train_batch_size=3 \
  --adam_beta1=0.95 \
  --adam_beta2=0.99 \
  --average_tokens_across_devices=True \
  --copy_paste_prob=0.2 \
  --dataloader_num_workers=4 \
  --dataloader_persistent_workers=True \
  --dataloader_pin_memory=True \
  --ddp_find_unused_parameters=True \
  --early_stopping=25 \
  --fold=0 \
  --learning_rate=1e-4 \
  --max_grad_norm=3 \
  --mixup_prob=0.2 \
  --num_train_epochs=75 \
  --random_erase_prob=0.2 \
  --use_6_classes=True \
  --use_cross_entropy_loss=True \
  --use_instance_crops=True \
  --use_random_crops=True \
  --use_stride4=False \
  --use_stride2=True \
  --warmup_steps=64 \
  --x_rotation_limit=10 \
  --y_rotation_limit=10
done
