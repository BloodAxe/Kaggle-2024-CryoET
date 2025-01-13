torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=hrnet  \
--per_device_train_batch_size=6 --learning_rate=5e-5 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=15 --x_rotation_limit=15  \
--num_train_epochs=150 --early_stopping=50 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=0  --max_grad_norm=3 --ddp_find_unused_parameters=True

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=hrnet  \
--per_device_train_batch_size=6 --learning_rate=5e-5 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=15 --x_rotation_limit=15  \
--num_train_epochs=150 --early_stopping=50 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=1  --max_grad_norm=3 --ddp_find_unused_parameters=True

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=hrnet  \
--per_device_train_batch_size=6 --learning_rate=5e-5 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=15 --x_rotation_limit=15  \
--num_train_epochs=150 --early_stopping=50 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2  --max_grad_norm=3 --ddp_find_unused_parameters=True

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=hrnet  \
--per_device_train_batch_size=6 --learning_rate=5e-5 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=15 --x_rotation_limit=15  \
--num_train_epochs=150 --early_stopping=50 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=3  --max_grad_norm=3 --ddp_find_unused_parameters=True

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=hrnet  \
--per_device_train_batch_size=6 --learning_rate=5e-5 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=15 --x_rotation_limit=15  \
--num_train_epochs=150 --early_stopping=50 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=4  --max_grad_norm=3 --ddp_find_unused_parameters=True