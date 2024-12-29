torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=16 --learning_rate=1e-3 --optim=sgd \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=0 --ema --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp

torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=16 --learning_rate=1e-3 --optim=sgd \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=1 --ema --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp

torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=16 --learning_rate=1e-3 --optim=sgd \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=2 --ema --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp

torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=16 --learning_rate=1e-3 --optim=sgd \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=3 --ema --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp

torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet \
--pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=16 --learning_rate=1e-3 --optim=sgd \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
--fold=4 --ema --train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp