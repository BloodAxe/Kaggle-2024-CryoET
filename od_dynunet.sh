torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=dynunet --window_size=128 --num_crops_per_study=256 \
--per_device_train_batch_size=6 --learning_rate=3e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=0 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25 --apply_loss_on_each_stride=True --random_erase_prob=0.05 --gaussian_noise_sigma=0.00001

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=dynunet --window_size=128 --num_crops_per_study=256 \
--per_device_train_batch_size=6 --learning_rate=3e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=1 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25 --apply_loss_on_each_stride=True --random_erase_prob=0.05 --gaussian_noise_sigma=0.00001

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=dynunet --window_size=128 --num_crops_per_study=256 \
--per_device_train_batch_size=6 --learning_rate=3e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25 --apply_loss_on_each_stride=True --random_erase_prob=0.05 --gaussian_noise_sigma=0.00001

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=dynunet --window_size=128 --num_crops_per_study=256 \
--per_device_train_batch_size=6 --learning_rate=3e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=3 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25 --apply_loss_on_each_stride=True --random_erase_prob=0.05 --gaussian_noise_sigma=0.00001

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=dynunet --window_size=128 --num_crops_per_study=256 \
--per_device_train_batch_size=6 --learning_rate=3e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=4 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25 --apply_loss_on_each_stride=True --random_erase_prob=0.05 --gaussian_noise_sigma=0.00001
