python train.py --name stanford_256_g8 \
       --dataroot datasets/stanford_campus_dataset/ \
       --dataset_mode stanford \
       --loadSize 256 --ngf 64 \
       --input_nc 6 \
       --num_D 2 \
       --no_first_img \
       --fineSize 256 \
       --gpu_ids 0,1,2,3,4,5,6,7 --n_gpus_gen 6 \
       --n_frames_total 12 \
       --niter 20 --niter_decay 20 \
       --batchSize 8 \
       --tf_log

