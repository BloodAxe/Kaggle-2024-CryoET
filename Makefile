tensorboard-daemon:
	nohup tensorboard --logdir=./runs --host 0.0.0.0 --port 6006 > /dev/null 2>&1 &

