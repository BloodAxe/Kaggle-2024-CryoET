torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=False  --use_instance_crops=True --use_random_crops=True \
--fold=0 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_centernet_nms=True

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=False  --use_instance_crops=True --use_random_crops=True \
--fold=0 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_centernet_nms=True --use_l1_loss=True

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=False  --use_instance_crops=True --use_random_crops=True \
--fold=0 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_centernet_nms=False

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=False  --use_instance_crops=True --use_random_crops=True \
--fold=0 --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_centernet_nms=True

torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=segresnetv2 \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=False  --use_instance_crops=True --use_random_crops=True \
--fold=0 --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25 --use_centernet_nms=False

