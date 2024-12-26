tensorboard-daemon:
	nohup tensorboard --logdir=./runs --host 0.0.0.0 --port 6006 > /dev/null 2>&1 &

unetr_baseline:
	torchrun --standalone --nproc-per-node=4 train.py --per_device_train_batch_size=6 \
    --dataloader_num_workers=4 --dataloader_persistent_workers=True --dataloader_pin_memory=True \
    --num_train_epochs=100 --warmup_steps=32