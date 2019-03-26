python train.py --name kaist_256_g1_ir2rgb \
       --dataroot /home/ubuntu/datasets/images/ \
       --input_nc 3 \
       --output_nc 3 \
       --loadSize 256 --ngf 64 \
       --max_frames_per_gpu 6 \
       --no_first_img \
       --n_frames_total 12 \
       --niter 20 --niter_decay 20
