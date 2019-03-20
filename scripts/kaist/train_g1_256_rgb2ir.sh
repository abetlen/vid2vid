python train.py --name stanford_256_g1_rgb2ir \
       --dataroot datasets/stanford_campus_dataset/ \
       --dataset_mode kaist \
       --loadSize 256 --ngf 64 \
       --input_nc 3 \
       --output_nc 1 \
       --imgmode rgb2ir \
       --max_frames_per_gpu 6 --n_frames_total 12 \
       --niter 20 --niter_decay 20 \
       --tf_log
