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

unetr_baseline_instance_and_random_crops_fold_0:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=0

segresnet_baseline_instance_and_random_crops_fold_0:
	torchrun --standalone --nproc-per-node=4 train.py \
    --model_name=segresnet \
    --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=0

unetr_baseline_instance_and_random_crops_fold_1:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=1

unetr_baseline_instance_and_random_crops_fold_2:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=2

unetr_baseline_instance_and_random_crops_fold_3:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=3

unetr_baseline_instance_and_random_crops_fold_4:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=4

unetr_baseline_all_folds: unetr_baseline_instance_and_random_crops_fold_0 unetr_baseline_instance_and_random_crops_fold_1 unetr_baseline_instance_and_random_crops_fold_2 unetr_baseline_instance_and_random_crops_fold_3 unetr_baseline_instance_and_random_crops_fold_4

unetr_baseline_all_crops:
	torchrun --standalone --nproc-per-node=4 train.py \
    --pretrained_backbone_path=pretrained/swin_unetr_btcv_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_sliding_crops=True --use_instance_crops=True --use_random_crops=True


unetr_all: unetr_baseline_use_random_crops unetr_baseline_use_instance_crops unetr_baseline_instance_and_random_crops unetr_baseline_all_crops
