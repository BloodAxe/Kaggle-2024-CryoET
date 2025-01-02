torchrun --standalone --nproc-per-node=2 train_od.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=4 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=64 --average_tokens_across_devices=True  \
--fold=0 --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --max_grad_norm=3  --use_random_crops=True --use_instance_crops=True --ddp_find_unused_parameters=True

torchrun --standalone --nproc-per-node=2 train_od.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=2 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=64 --average_tokens_across_devices=False  --use_instance_crops=True --use_random_crops=True \
--fold=1 --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --max_grad_norm=3

torchrun --standalone --nproc-per-node=2 train_od.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=2 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=64 --average_tokens_across_devices=False  --use_instance_crops=True --use_random_crops=True \
--fold=2 --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --max_grad_norm=3

torchrun --standalone --nproc-per-node=2 train_od.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=2 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=64 --average_tokens_across_devices=False  --use_instance_crops=True --use_random_crops=True \
--fold=3 --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --max_grad_norm=3

torchrun --standalone --nproc-per-node=2 train_od.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=2 --learning_rate=1e-5 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=64 --average_tokens_across_devices=False  --use_instance_crops=True --use_random_crops=True \
--fold=4 --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --max_grad_norm=3