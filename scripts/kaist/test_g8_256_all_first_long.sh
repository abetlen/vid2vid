#!/bin/bash

if [[ $# -eq 1 ]]
then
    START_FRAME=$1
else
    START_FRAME=0
fi

python test.py --name kaist_256_g8_ir2rgb \
       --dataroot /home/ubuntu/datasets/images/ \
       --dataset_mode kaist_test_single \
       --loadSize 512 --ngf 64 \
       --fineSize 256 \
       --input_nc 3 \
       --output_nc 3 \
       --no_first_img \
       --imgmode ir2rgb \
       --how_many 60000 \
       --start_frame $START_FRAME
