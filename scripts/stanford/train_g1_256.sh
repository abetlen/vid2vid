python train.py --name stanford_256_g1 \
--dataroot datasets/stanford_campus_dataset/ --dataset_mode stanford \
--input_nc 6 --loadSize 256 --ngf 64 \
--max_frames_per_gpu 6 --n_frames_total 12 \
--niter 20 --niter_decay 20
