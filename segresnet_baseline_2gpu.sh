torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --window_size=128 --num_crops_per_study=128 \
--learning_rate=1e-5 --adam_beta1=0.95 --adam_beta2=0.99 --ema \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --early_stopping=10 --warmup_steps=64 --average_tokens_across_devices=True \
--use_instance_crops=True --use_random_crops=True \
--train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --fold=0

torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --window_size=128 --num_crops_per_study=128 \
--learning_rate=1e-5 --adam_beta1=0.95 --adam_beta2=0.99 --ema \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --early_stopping=10 --warmup_steps=64 --average_tokens_across_devices=True \
--use_instance_crops=True --use_random_crops=True \
--train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --fold=1

torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --window_size=128 --num_crops_per_study=128 \
--learning_rate=1e-5 --adam_beta1=0.95 --adam_beta2=0.99 --ema \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --early_stopping=10 --warmup_steps=64 --average_tokens_across_devices=True \
--use_instance_crops=True --use_random_crops=True \
--train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --fold=2

torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --window_size=128 --num_crops_per_study=128 \
--learning_rate=1e-5 --adam_beta1=0.95 --adam_beta2=0.99 --ema \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --early_stopping=10 --warmup_steps=64 --average_tokens_across_devices=True \
--use_instance_crops=True --use_random_crops=True \
--train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --fold=3

torchrun --standalone --nproc-per-node=2 train.py \
--model_name=segresnet --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
--per_device_train_batch_size=6 --window_size=128 --num_crops_per_study=128 \
--learning_rate=1e-5 --adam_beta1=0.95 --adam_beta2=0.99 --ema \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=50 --early_stopping=10 --warmup_steps=64 --average_tokens_across_devices=True \
--use_instance_crops=True --use_random_crops=True \
--train_modes=denoised,ctfdeconvolved,isonetcorrected,wbp --fold=4