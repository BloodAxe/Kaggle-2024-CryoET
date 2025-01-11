torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=hrnet --num_crops_per_study=256 \
--per_device_train_batch_size=16 --learning_rate=3e-4 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=15 --x_rotation_limit=15 --use_centernet_nms=False \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=0 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --random_erase_prob=0.05
