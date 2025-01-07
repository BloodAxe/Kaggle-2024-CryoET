torchrun --standalone --nproc-per-node=4 train_od.py \
--model_name=unetr --window_size=96 --num_crops_per_study=256 \
--pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
--per_device_train_batch_size=4 --learning_rate=1e-4 \
--adam_beta1=0.95 --adam_beta2=0.99 \
--dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
--y_rotation_limit=10 --x_rotation_limit=10 \
--num_train_epochs=150 --warmup_steps=64 --average_tokens_across_devices=True  --use_instance_crops=True --use_random_crops=True \
--fold=0 --train_modes=denoised --max_grad_norm=3 --ddp_find_unused_parameters=True --early_stopping=25
