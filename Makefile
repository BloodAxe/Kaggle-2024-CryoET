tensorboard-daemon:
	nohup tensorboard --logdir=./runs --host 0.0.0.0 --port 6006 > /dev/null 2>&1 &

unetr_baseline_use_sliding_crops:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_sliding_crops=True

unetr_baseline_use_random_crops:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_random_crops=True

unetr_baseline_use_instance_crops:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True

unetr_baseline_instance_and_random_crops:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True

unetr_baseline_all_crops:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_sliding_crops=True --use_instance_crops=True --use_random_crops=True


unetr_all: unetr_baseline_use_sliding_crops unetr_baseline_use_random_crops unetr_baseline_use_instance_crops unetr_baseline_instance_and_random_crops unetr_baseline_all_crops
