### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import random

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle

# data_loader = CreateDataLoader(opt)
# dataset = data_loader.load_data()
from patagona_common.data.datasets import KAISTTemporalDataset
dataset = KAISTTemporalDataset(root_dir='/home/ubuntu/datasets', video_sets=None)
model = create_model(opt)
visualizer = Visualizer(opt)
input_nc = 1 if opt.label_nc != 0 else opt.input_nc

save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
print('Doing %d frames' % len(dataset))

start_idxs = list(range(len(dataset)))
random.shuffle(start_idxs)

for i in range(opt.how_many):

    idx = start_idxs[i]
    data = dataset[idx]
    model.fake_B_prev = None
    data['A'] = data['A'].unsqueeze(0)

    _, _, height, width = data['A'].size()
    A = Variable(data['A']).view(1, -1, input_nc, height, width)
    # B = Variable(data['B']).view(1, -1, opt.output_nc, height, width) if len(data['B'].size()) > 2 else None
    B = None
    inst = None

    for i in range(opt.n_frames_total):
        generated = model.inference(A[0, i, ...], B, inst)

        c = 3 if opt.input_nc == 3 else 1
        real_A = util.tensor2im(generated[1][:c], normalize=True)
        fake_B = util.tensor2im(generated[0].data[0])
        visual_list = [('real_A', real_A), 
                       ('fake_B', util.tensor2im(generated[0].data[0]))]
    visuals = OrderedDict(visual_list) 
    img_path = data['A_path'] + f'_{i}'
    print('process image... %s' % img_path)
    visualizer.save_images(save_dir, visuals, img_path)
