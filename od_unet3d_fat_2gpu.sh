torchrun --standalone --nproc-per-node=2 train_od.py \
--model_name=unet3d-fat \
--per_device_train_batch_size=16 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--num_train_epochs=100 --warmup_steps=64 --average_tokens_across_devices=True   --use_random_crops=True --use_instance_crops=True --ddp_find_unused_parameters=True \
--fold=0 --train_modes=ctfdeconvolved,isonetcorrected,wbp --max_grad_norm=3  --early_stopping=25
