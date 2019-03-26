python test_custom.py --name kaist_256_g1_ir2rgb \
       --dataroot /home/ubuntu/datasets/images/ \
       --dataset_mode kaist \
       --loadSize 256 --ngf 64 \
       --fineSize 256 \
       --input_nc 3 \
       --output_nc 3 \
       --no_first_img \
       --n_frames_total 12 \
       --imgmode ir2rgb \
       --how_many 2 \

