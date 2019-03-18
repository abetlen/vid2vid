import os
import pandas as pd
import numpy as np
#import pims

from PIL import Image, ImageFilter, ImageDraw
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from patagona_common.image import gaussian2D

from collections import OrderedDict

from datasets.regions import regions_deathCircle_v3 as regions
# scene: string -- scene to use
# video: int -- video id
# anchor: (x, y) -- Top left corner coordinates
# size: (w, h) -- width and height of region
# start: first frame to use
# end: last frame to use

colors = {'background': '#000000',
          'Biker': '#FF0000',
          'Bus': '#00FF00',
          'Car': '#00FFFF',
          'Cart': '#FF0000',
          'Pedestrian': '#0000FF',
          'Skater': '#0000FF'}

labels = {'Biker': 0,
          'Bus': 1,
          'Car': 2,
          'Cart': 3,
          'Pedestrian': 4,
          'Skater': 5}

# (scene, n_videos)
scenes = [('bookstore', 7),
          ('coupa', 4),
          ('deathCircle', 5),
          ('gates', 9),
          ('hyang', 15),
          ('little', 4),
          ('nexus', 10),
          ('quad', 4)]

classes_of_interest = ['Biker',
                       'Cart',
                       'Pedestrian',
                       'Skater']


def bboxshow(ax, annotations, label_colors=colors):
    """Plots pandas annotation bounding boxes on a matplotlib axis
    """
    for index, anno in annotations.iterrows():
        x = anno['xmin']
        y = anno['ymin']
        w = anno['xmax'] - x
        h = anno['ymax'] - y
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=label_colors[anno['label']], facecolor='none')
        ax.add_patch(rect)


def centershow(ax, annotations, label_colors=colors):
    """Plots centers on axis
    """
    for index, anno in annotations.iterrows():
        xcenter = anno['xcenter']
        ycenter = anno['ycenter']
        ax.scatter([xcenter], [ycenter], color=label_colors[anno['label']])


def regionboxshow(ax, anchor, size, color='c'):
        (x, y) = anchor
        (w, h) = size

        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')

        ax.add_patch(rect)


def sampleshow(sample, region_bounds=False):
    annotations = sample['annotations']
    frame = sample['frame']
    background = sample['background']
    mask = sample['mask']

    if isinstance(frame, torch.Tensor):
        frame = (frame * 0.5) + 0.5
        frame = frame.permute(1, 2, 0).numpy()
    if isinstance(background, torch.Tensor):
        background = (background * 0.5) + 0.5
        background = background.permute(1, 2, 0).numpy()
    if isinstance(mask, torch.Tensor):
        mask = (mask * 0.5) + 0.5
        mask = mask.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(frame)
    ax[1].imshow(background)
    ax[2].imshow(mask)

    bboxshow(ax[0], annotations[annotations['lost'] == 0])
    bboxshow(ax[1], annotations[annotations['lost'] == 0])

    regionboxshow(ax[0], sample['region']['anchor'], sample['region']['size'])
    regionboxshow(ax[1], sample['region']['anchor'], sample['region']['size'])
    plt.show()


def make_video_sequence(dataset, filename='output.mp4', centercoords=True):
    assert type(dataset[0]['frame']) is not torch.Tensor, 'type cannot be torch.Tensor'

    videodims = dataset[0]['region']['size']
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(filename, fourcc, 25, videodims)
    totalframes = len(dataset)
    framecount = 0
    for sample in dataset:
        frame = sample['frame']
        region = sample['region']
        annotations = sample['annotations']

        if centercoords:
            xcenter = np.asarray(annotations['xcenter'])
            ycenter = np.asarray(annotations['ycenter'])

            # Draw some crosses
            points = list(zip(xcenter, ycenter)) \
                + list(zip(xcenter + 1, ycenter)) \
                + list(zip(xcenter, ycenter + 1)) \
                + list(zip(xcenter - 1, ycenter)) \
                + list(zip(xcenter, ycenter - 1))

            draw = ImageDraw.Draw(frame)
            draw.text((0, 0), f"s:{region['scene']} v:{region['video']} r:{region['rid']} f:{sample['frame_idx']}")
            draw.point(points, fill='#FF0000')
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        framecount += 1
        print(f'{framecount}/{totalframes} complete')
    video.release()


class GaussianBlur(object):
    def __init__(self, sigma=2):
        self.sigma = sigma

    def __call__(self, image):
        try:
            image = image.filter(ImageFilter.GaussianBlur(self.sigma))
        except TypeError:
            print('Must be a PIL Image')
        return image


class StanfordDataset(Dataset):
    def __init__(self, rootdir='datasets/stanford_campus_dataset',
                 regions=regions,
                 transform_list=[GaussianBlur(sigma=1.5)],
                 normalize=True,
                 to_tensor=True):
        self.rootdir = rootdir

        if transform_list is None:
            transform_list = []
        final_transforms = []

        self.to_tensor = to_tensor
        if to_tensor:
            final_transforms.append(transforms.ToTensor())
        if normalize:
            final_transforms.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)))

        self.transform = transforms.Compose(transform_list)
        self.final_transform = transforms.Compose(final_transforms)

        self.n_frames = 0
        self.n_prev_frames = 2

        fileset = set()
        frame_root_path = {}
        annotations = {}
        background_path = {}

        if type(regions) == list:
            self.data = OrderedDict()
            for region in regions:
                fileset.add((region['scene'], region['video']))
            for f in fileset:
                video_path = os.path.join(rootdir, 'videos', f[0],
                                          'video' + str(f[1]))
                anno_path = os.path.join(rootdir, 'annotations', f[0],
                                         'video' + str(f[1]), 'annotations_with_centers.txt')

                frame_root_path[f] = video_path
                annotations[f] = pd.read_csv(anno_path,
                                             delimiter=' ',
                                             header=None,
                                             names=['id', 'xmin', 'ymin',
                                                    'xmax', 'ymax', 'frame',
                                                    'lost', 'occluded',
                                                    'generated', 'label',
                                                    'xcenter', 'ycenter'])
                background_path[f] = video_path

            for region in regions:
                f = (region['scene'], region['video'])
                rid = region['rid']
                background = Image.open(
                    os.path.join(background_path[f],
                                 f'r{rid}_background.png'))
                background.load()

                for (start, end) in region['seq']:
                    reg = {'scene': region['scene'],
                           'video': region['video'],
                           'rid': region['rid'],
                           'anchor': region['anchor'],
                           'size': region['size'],
                           'start': start,
                           'end': end}

                    self.data[self.n_frames] = {'region': reg,
                                                'rootdir': frame_root_path[f],
                                                'prefix': f'r{rid}_frame',
                                                'background': background,
                                                'annotations': annotations[f]}
                    self.n_frames += (end - start - self.n_prev_frames)

        elif regions == 'all':
            raise NotImplementedError('Currently only supports lists of Regions')
        else:
            raise NotImplementedError('Currently only supports lists of Regions')

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):

        if idx >= len(self):
            raise IndexError()
        idxs = self.data.keys()

        start_idx = max([i for i in idxs if i <= idx])

        data = self.data[start_idx]
        frame_idx = (idx - start_idx + self.n_prev_frames) + data['region']['start']

        # frame = Image.fromarray(data['video'][frame_idx], mode='RGB')

        with open(os.path.join(data['rootdir'], data['prefix'] + str(frame_idx + 1).zfill(5) + '.jpg'), 'rb') as fd:
            frame = Image.open(fd)
            frame.load()
        with open(os.path.join(data['rootdir'], data['prefix'] + str(frame_idx).zfill(5) + '.jpg'), 'rb') as fd:
            prev_frame = Image.open(fd)
            prev_frame.load()
        with open(os.path.join(data['rootdir'], data['prefix'] + str(frame_idx - 1).zfill(5) + '.jpg'), 'rb') as fd:
            prevprev_frame = Image.open(fd)
            prevprev_frame.load()

        curr_annotations = data['annotations'][data['annotations']['frame'] == frame_idx]
        prev_annotations = data['annotations'][data['annotations']['frame'] == (frame_idx - 1)]
        # prevprev_annotations = data['annotations'][data['annotations']['frame'] == (frame_idx - 2)]
        sample = {'frame': frame,
                  'prev_frame': prev_frame,
                  'prevprev_frame': prevprev_frame,
                  'background': data['background'],
                  'region': data['region'],
                  'annotations': curr_annotations,
                  'frame_idx': frame_idx}

        if sample['region']['anchor'] is not None:
            sample = self.region_crop(sample)

        if sample['region']['anchor'] is None:
            raise NotImplementedError('Only static regions supported')

        sample['uv'] = self.generate_bbox_flow_field(sample, curr_annotations, prev_annotations)

        sample['frame'] = self.transform(sample['frame'])
        sample['prev_frame'] = self.transform(sample['prev_frame'])
        sample['prevprev_frame'] = self.transform(sample['prevprev_frame'])
        sample['background'] = self.transform(sample['background'])

        sample['mask'] = self.generate_label_mask(sample)

        sample['frame'] = self.final_transform(sample['frame'])
        sample['prev_frame'] = self.final_transform(sample['prev_frame'])
        sample['prevprev_frame'] = self.final_transform(sample['prevprev_frame'])

        sample['background'] = self.final_transform(sample['background'])

        sample['mask'] = self.final_transform(sample['mask'])

        sample['annotations'].loc[:, 'label'] = sample['annotations']['label'].map(labels)

        if self.to_tensor:
            # sample['annotations'] = torch.Tensor(sample['annotations'].values)
            sample['annotations'] = 0
        # need to write own collate fn
        return sample
        # return {'frame': frame,
        #         'mask': mask,
        #         'background': background,
        #         'region': region,
        #         'annotations': annotations,
        #         'frame_idx': frame_idx,}

    @staticmethod
    def generate_bbox_flow_field(sample, curr_annotations, prev_annotations, label_width=16, normalize=False):
        curr = curr_annotations[curr_annotations.id.isin(sample['annotations'].id)]
        prev = prev_annotations[prev_annotations.id.isin(sample['annotations'].id)]

        curr = curr.set_index('id')
        prev = prev.set_index('id')
        sam = sample['annotations'].set_index('id')

        sam['u'] = curr['xcenter'] - prev['xcenter']
        sam['v'] = curr['ycenter'] - prev['ycenter']

        dim = sample['region']['size']
        u = torch.zeros(dim)
        v = torch.zeros(dim)

        for idx, row in sam[sam['lost'] == 0].iterrows():
            if row['label'] in classes_of_interest:
                xmin = row['xmin']
                ymin = row['ymin']

                xcenter = row['xcenter']
                ycenter = row['ycenter']

                if np.isnan(xcenter):
                    xcenter = (row['xmax'] - xmin) / 2.
                if np.isnan(ycenter):
                    ycenter = (row['ymax'] - ymin) / 2.

                xcenter = int(xcenter)
                ycenter = int(ycenter)

                half = int(label_width / 2)

                xmin = xcenter - half
                ymin = ycenter - half
                xmax = xcenter + half
                ymax = ycenter + half

                if row['u'] != 0 or row['v'] != 0:
                    magnitude = math.sqrt(row['u']**2 + row['v']**2)
                    if normalize:
                        u_norm = row['u'] / magnitude
                        v_norm = row['v'] / magnitude
                    else:
                        u_norm = row['u']
                        v_norm = row['v']

                    if np.isnan(u_norm):
                        u_norm = 0
                    if np.isnan(v_norm):
                        v_norm = 0

                    u[ymin:ymax, xmin:xmax] = u_norm
                    v[ymin:ymax, xmin:xmax] = v_norm

                    # u[ymin:ymax, xmin:xmax] = row['u']
                    # v[ymin:ymax, xmin:xmax] = row['v']

        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
        return torch.cat((u, v), 0)

    @staticmethod
    def generate_flow_field_image(flow_field):
        flow_field = flow_field.squeeze()
        flow = flow_field.permute(1, 2, 0).numpy()

        hsv = np.zeros(flow[:, :, 0].shape + (3,), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        bgr = bgr / 255

        rgb = torch.Tensor(bgr[:, :, np.argsort([2, 1, 0])])
        rgb = rgb.permute(2, 0, 1)
        return rgb

    @staticmethod
    def draw_properties_img(sample, res=128):
        im = Image.new(mode='RGB', size=(res, res), color='#000000')

        scene = str(sample['region']['scene'][0])
        video = sample['region']['video'].numpy()[0]
        rid = sample['region']['rid'].numpy()[0]
        frame = sample['frame_idx'].numpy()[0] + 1

        draw = ImageDraw.Draw(im)

        draw.text((10, 10), f'Scene: {scene}')
        draw.text((10, 20), f'Video: {video}')
        draw.text((10, 30), f'RID: {rid}')
        draw.text((10, 40), f'frame: {frame}')

        return transforms.ToTensor()(im)

    @staticmethod
    def generate_label_mask(sample, label_width=32, mask_shape='gaussian'):
        annotations = sample['annotations']
        size = sample['region']['size']

        mask = Image.new(mode='RGB', size=size, color=colors['background'])
        draw = ImageDraw.Draw(mask)

        if mask_shape == 'gaussian':
            gauss_mask = gaussian2D(label_width)

        for index, anno in annotations[annotations['lost'] == 0].iterrows():
            if anno['label'] in classes_of_interest:
                xcenter = anno['xcenter']
                ycenter = anno['ycenter']

                if np.isnan(xcenter):
                    xcenter = (anno['xmax'] - anno['xmin']) / 2.
                if np.isnan(ycenter):
                    ycenter = (anno['ymax'] - anno['ymin']) / 2.

                xcenter = int(xcenter)
                ycenter = int(ycenter)

                half = label_width / 2
                if mask_shape == 'circle':
                    draw.ellipse([xcenter - half, ycenter - half,
                                  xcenter + half - 1, ycenter + half - 1],
                                 fill=colors[anno['label']])
                elif mask_shape == 'square':
                    draw.rectangle([xcenter - half, ycenter - half,
                                    xcenter + half - 1, ycenter + half - 1],
                                   fill=colors[anno['label']])
                elif mask_shape == 'gaussian':
                    draw.bitmap((xcenter - half, ycenter - half), gauss_mask, fill=colors[anno['label']])

                elif mask_shape == 'point':
                    draw.point([xcenter, ycenter],
                               fill=colors[anno['label']])

        return mask

    @staticmethod
    def region_crop(sample):
        (x, y) = sample['region']['anchor']
        (w, h) = sample['region']['size']

        background = sample['background']
        annos = sample['annotations']

        # Crop frame
        # frame = frame.crop((x, y, x + w, y + h))
        # background = background.crop((x, y, x + w, y + h))
        # Crop Annotations
        # Filter out-of-region
        # annos = annos[~((annos['xmax'] <= x) | (annos['xmin'] >= (x + w)) | (annos['ymax'] >= (y + h)) | (annos['ymin'] <= y))].copy()

        annos = annos[~((annos['xmax'] < x)
                        | (annos['ymax'] < y)
                        | (annos['xmin'] > (x + w))
                        | (annos['ymin'] > (y + h)))].copy()

        # Crop bboxes
        annos.loc[:, 'xmin'] = annos['xmin'] - x
        annos.loc[:, 'xmax'] = annos['xmax'] - x
        annos.loc[:, 'ymin'] = annos['ymin'] - y
        annos.loc[:, 'ymax'] = annos['ymax'] - y

        annos.loc[:, 'xcenter'] = annos['xcenter'] - x
        annos.loc[:, 'ycenter'] = annos['ycenter'] - y

        annos.loc[annos['xmin'] < 0, 'xmin'] = 0
        annos.loc[annos['ymin'] < 0, 'ymin'] = 0
        annos.loc[annos['xmax'] > w, 'xmax'] = w
        annos.loc[annos['ymax'] > h, 'ymax'] = h

        sample['background'] = background
        sample['annotations'] = annos

        return sample

    @staticmethod
    def show_frame_with_bboxes(self, scene, video, frame_idx, rootdir='datasets/stanford_campus_dataset'):

        video_path = os.path.join(rootdir, 'videos', scene,
                                  f'video{video}', 'video.mov')
        v = pims.Video(video_path)

        frame = v[frame_idx]

        fix, ax = plt.subplots(1)
        ax.imshow(frame)
        anno_path = os.path.join(rootdir, 'annotations', scene,
                                 f'video{video}',
                                 'annotations.txt')

        annotations = pd.read_csv(anno_path,
                                  delimiter=' ',
                                  header=None,
                                  names=['id', 'xmin', 'ymin',
                                         'xmax', 'ymax', 'frame',
                                         'lost', 'occluded',
                                         'generated', 'label',
                                         'xcenter', 'ycenter'])
        anno = annotations[(annotations['frame'] == frame_idx) & (annotations['lost'] == 0)]

        self.plot_bboxes(ax, anno)
        plt.show()


n_one_hot = 3
one_hot_idxs = {'background': 0,
                'Biker': 1,
                'Cart': 1,
                'Pedestrian': 2,
                'Skater': 2}


class StanfordDatasetTemporal(Dataset):
    def __init__(self, rootdir='/home/ubuntu/datasets',
                 regions=regions,
                 transform_list=[GaussianBlur(sigma=1.5)],
                 normalize=True,
                 to_tensor=True,
                 n_classes=n_one_hot,
                 one_hot_idxs=one_hot_idxs,
                 n_sequential_frames=12,
                 classes_of_interest=classes_of_interest):
        self.rootdir = rootdir

        if transform_list is None:
            transform_list = []
        final_transforms = []

        self.to_tensor = to_tensor
        if to_tensor:
            final_transforms.append(transforms.ToTensor())
        if normalize:
            final_transforms.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5)))

        self.transform = transforms.Compose(transform_list)
        self.final_transform = transforms.Compose(final_transforms)

        self.total_frames = 0
        self.n_seq_frames = n_sequential_frames
        self.n_classes = n_classes
        self.one_hot_idxs = one_hot_idxs
        self.classes_of_interest = classes_of_interest

        fileset = set()
        frame_root_path = {}
        annotations = {}
        background_path = {}

        if type(regions) == list:
            self.data = OrderedDict()
            for region in regions:
                fileset.add((region['scene'], region['video']))
            for f in fileset:
                video_path = os.path.join(rootdir, 'videos', f[0],
                                          'video' + str(f[1]))
                anno_path = os.path.join(rootdir, 'annotations', f[0],
                                         'video' + str(f[1]), 'annotations_with_centers.txt')

                frame_root_path[f] = video_path
                annotations[f] = pd.read_csv(anno_path,
                                             delimiter=' ',
                                             header=None,
                                             names=['id', 'xmin', 'ymin',
                                                    'xmax', 'ymax', 'frame',
                                                    'lost', 'occluded',
                                                    'generated', 'label',
                                                    'xcenter', 'ycenter'])
                background_path[f] = video_path

            for region in regions:
                f = (region['scene'], region['video'])
                rid = region['rid']
                background = Image.open(
                    os.path.join(background_path[f],
                                 f'r{rid}_background.png'))
                background.load()

                for (start, end) in region['seq']:
                    reg = {'scene': region['scene'],
                           'video': region['video'],
                           'rid': region['rid'],
                           'anchor': region['anchor'],
                           'size': region['size'],
                           'start': start,
                           'end': end}

                    self.data[self.total_frames] = {'region': reg,
                                                    'rootdir': frame_root_path[f],
                                                    'prefix': f'r{rid}_frame',
                                                    'background': background,
                                                    'annotations': annotations[f]}
                    self.total_frames += (end - start - self.n_seq_frames + 1)

        elif regions == 'all':
            raise NotImplementedError('Currently only supports lists of Regions')
        else:
            raise NotImplementedError('Currently only supports lists of Regions')

    def initialize(self, opt):
        pass

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError()
        idxs = self.data.keys()

        start_idx = max([i for i in idxs if i <= idx])

        data = self.data[start_idx]
        width, height = data['region']['size']
        background = self.final_transform(self.transform(data['background']))
        background = background.unsqueeze(0)

        first_frame_idx = (idx - start_idx) + data['region']['start']
        A = torch.empty([self.n_seq_frames, self.n_classes + 3, width, height])
        B = torch.empty([self.n_seq_frames, 3, width, height])

        inst, A_path, B_path = 0, 0, 0

        #  Build up tensor
        for i in range(self.n_seq_frames):
            curr_frame_idx = first_frame_idx + i
            curr_img_file_idx = curr_frame_idx + 1

            # A tensor
            annotation = data['annotations'][data['annotations']['frame'] == curr_frame_idx]
            annotation = self.crop_annotation(annotation, data['region'])

            one_hot_tensor = self.generate_one_hot_tensor(annotation, dim, 16, self.n_classes, self.one_hot_idxs, self.classes_of_interest)
            A[i] = torch.cat([one_hot_tensor, background], 1)

            # B tensor
            with open(os.path.join(data['rootdir'], data['prefix'] + str(curr_img_file_idx).zfill(5) + '.jpg'), 'rb') as fd:
                frame = Image.open(fd)
                frame.load()
                frame_tensor = self.final_transform(self.transform(frame))
                B[i] = frame_tensor
        A = A.view(-1, width, height)
        B = B.view(-1, width, height)
        # B: expected output
        return {'A': A, 'B': B, 'inst': inst, 'A_path': A_path, 'B_paths': B_path}

    @staticmethod
    def generate_one_hot_tensor(annotation, dim, label_width, n_classes, one_hot_idxs, classes_of_interest):
        """Returns n_one_hotxHxW Tensor with N one-hot tensors (including background) for each label
        """
        W, H = dim
        one_hot_tensor = torch.zeros([n_classes, W, H])
        background_idx = one_hot_idxs['background']
        one_hot_tensor[background_idx] = 1

        for index, anno in annotation[annotation['lost'] == 0].iterrows():
            if anno['label'] in classes_of_interest:
                xcenter = anno['xcenter']
                ycenter = anno['ycenter']

                if np.isnan(xcenter):
                    xcenter = (anno['xmax'] - anno['xmin']) / 2.
                if np.isnan(ycenter):
                    ycenter = (anno['ymax'] - anno['ymin']) / 2.

                xcenter = int(xcenter)
                ycenter = int(ycenter)

                half = label_width // 2

                xmin = xcenter - half
                ymin = ycenter - half

                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0

                one_hot_idx = one_hot_idxs[anno['label']]
                one_hot_tensor[one_hot_idx,
                               ymin: ycenter + half,
                               xmin: xcenter + half] = 1
                one_hot_tensor[background_idx,
                               ymin: ycenter + half,
                               xmin: xcenter + half] = 0

        one_hot_tensor = one_hot_tensor.unsqueeze(0)
        return one_hot_tensor

    @staticmethod
    def crop_annotation(annotation, region):
        (x, y) = region['anchor']
        (w, h) = region['size']

        annotation = annotation[~((annotation['xmax'] < x)
                        | (annotation['ymax'] < y)
                        | (annotation['xmin'] > (x + w))
                        | (annotation['ymin'] > (y + h)))].copy()

        # Crop bboxes
        annotation.loc[:, 'xmin'] = annotation['xmin'] - x
        annotation.loc[:, 'xmax'] = annotation['xmax'] - x
        annotation.loc[:, 'ymin'] = annotation['ymin'] - y
        annotation.loc[:, 'ymax'] = annotation['ymax'] - y

        annotation.loc[:, 'xcenter'] = annotation['xcenter'] - x
        annotation.loc[:, 'ycenter'] = annotation['ycenter'] - y

        annotation.loc[annotation['xmin'] < 0, 'xmin'] = 0
        annotation.loc[annotation['ymin'] < 0, 'ymin'] = 0
        annotation.loc[annotation['xmax'] > w, 'xmax'] = w
        annotation.loc[annotation['ymax'] > h, 'ymax'] = h

        return annotation

    def name(self):
        return 'Stanford Campus'
