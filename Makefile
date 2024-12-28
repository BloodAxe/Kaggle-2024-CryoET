tensorboard-daemon:
	nohup tensorboard --logdir=./runs --host 0.0.0.0 --port 6006 > /dev/null 2>&1 &


segresnet_baseline_instance_and_random_crops_fold_0:
	torchrun --standalone --nproc-per-node=4 train.py \
    --model_name=segresnet \
    --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=0

segresnet_baseline_instance_and_random_crops_fold_1:
	torchrun --standalone --nproc-per-node=4 train.py \
    --model_name=segresnet \
    --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=1

segresnet_baseline_instance_and_random_crops_fold_2:
	torchrun --standalone --nproc-per-node=4 train.py \
    --model_name=segresnet \
    --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=2

segresnet_baseline_instance_and_random_crops_fold_3:
	torchrun --standalone --nproc-per-node=4 train.py \
    --model_name=segresnet \
    --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=3

segresnet_baseline_instance_and_random_crops_fold_4:
	torchrun --standalone --nproc-per-node=4 train.py \
    --model_name=segresnet \
    --pretrained_backbone_path=pretrained/wholeBody_ct_segmentation/models/model.pt \
    --per_device_train_batch_size=4 --learning_rate=1e-5 \
    --adam_beta1=0.95 --adam_beta2=0.99 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=50 --warmup_steps=32 --average_tokens_across_devices=True --use_instance_crops=True --use_random_crops=True \
    --fold=4

segresnet_baseline_all_folds: segresnet_baseline_instance_and_random_crops_fold_1 segresnet_baseline_instance_and_random_crops_fold_2 segresnet_baseline_instance_and_random_crops_fold_3 segresnet_baseline_instance_and_random_crops_fold_4

