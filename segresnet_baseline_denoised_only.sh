torchrun --standalone --nproc-per-node=4 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=0 --ema --train_modes=denoised

torchrun --standalone --nproc-per-node=4 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=1 --ema --train_modes=denoised

torchrun --standalone --nproc-per-node=4 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=2 --ema --train_modes=denoised

torchrun --standalone --nproc-per-node=4 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=3 --ema --train_modes=denoised

torchrun --standalone --nproc-per-node=4 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=4 --ema --train_modes=denoised