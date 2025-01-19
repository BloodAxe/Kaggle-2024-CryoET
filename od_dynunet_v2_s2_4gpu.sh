for FOLD in 0 1 2 3 4
do
  torchrun --standalone --nproc-per-node=4 train_od.py \
  --model_name=dynunet_v2  \
  --per_device_train_batch_size=8 --learning_rate=3e-4 \
  --adam_beta1=0.95 --adam_beta2=0.99 \
  --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
  --y_rotation_limit=10 --x_rotation_limit=10 \
  --num_train_epochs=75 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
  --fold=$FOLD  --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25 --use_stride4=False --use_6_classes=True \
  --copy_paste_prob=0.2 --random_erase_prob=0.2 --mixup_prob=0.2

done

