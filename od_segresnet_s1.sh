torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnet_s1  --ddp_find_unused_parameters=True \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=4  --max_grad_norm=3 --learning_rate=1e-4 --ema \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150  --early_stopping=25 --warmup_steps=64 --average_tokens_across_devices=True \
--train_modes=denoised --use_instance_crops=True --use_random_crops=True \
--random_erase_prob=0.1 --copy_paste_prob=0.5 --copy_paste_limit=2 \
--y_rotation_limit=20 --x_rotation_limit=20 \
--fold=0

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=1 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=3 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=4 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --ema --early_stopping=25
