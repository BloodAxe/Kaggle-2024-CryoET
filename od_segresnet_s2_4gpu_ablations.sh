# Ablate on hardest fold

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=4 --learning_rate=1e-4 --weight_decay=0.01 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2  --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_stride4=False

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=4 --learning_rate=1e-4 --weight_decay=0.01 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2  --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_stride4=False --use_offset_head=False

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=4 --learning_rate=1e-4 --weight_decay=0.01 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2  --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_stride4=False --use_6_classes=True

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=4 --learning_rate=1e-4 --weight_decay=0.01 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2  --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_stride4=False --use_6_classes=True --use_offset_head=False

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=4 --learning_rate=1e-4 --weight_decay=0.01 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2  --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_stride4=False --adam_beta1=0.95 --adam_beta2=0.99

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=4 --learning_rate=1e-4 --weight_decay=0.01 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2  --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_stride4=False --adam_beta1=0.9 --adam_beta2=0.95


torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=4 --learning_rate=1e-4 --weight_decay=0.01 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=2  --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_stride4=False --use_6_classes=True --use_offset_head=False --mixup_prob=0.5