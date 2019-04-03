import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'temporal':
        from data.temporal_dataset import TemporalDataset
        dataset = TemporalDataset()
    elif opt.dataset_mode == 'face':
        from data.face_dataset import FaceDataset
        dataset = FaceDataset()
    elif opt.dataset_mode == 'pose':
        from data.pose_dataset import PoseDataset
        dataset = PoseDataset()
    elif opt.dataset_mode == 'test':
        from data.test_dataset import TestDataset
        dataset = TestDataset()
    elif opt.dataset_mode == 'stanford':
        from patagona_common.data.datasets import StanfordDatasetTemporal
        dataset = StanfordDatasetTemporal(rootdir='/home/ubuntu/datasets')

    elif opt.dataset_mode == 'stanford_custom':
        from patagona_common.data.datasets import StanfordDatasetTemporal
        regions = [dict(scene='deathCircle', video=0, size=(1400, 1904), sequences=None, anchors=None)]
        dataset = StanfordDatasetTemporal(rootdir='/home/ubuntu/stanford_campus_dataset_synthetic',
                                          regions=regions,
                                          crop_mode='center',
                                          output_dim=(opt.loadSize, opt.loadSize),
                                          n_sequential_frames=opt.n_frames_G,
                                          inference=True)

    elif opt.dataset_mode == 'stanford_test':
        from patagona_common.data.datasets import StanfordDatasetTemporal
        regions = [dict(scene=opt.test_video_scene, video=test_video_id, size=(1400, 1904), sequences=None, anchors=None)]
        dataset = StanfordDatasetTemporal(rootdir='/home/ubuntu/datasets',
                                          regions=regions,
                                          crop_mode='center',
                                          output_dim=(opt.loadSize, opt.loadSize),
                                          n_sequential_frames=opt.n_frames_G,
                                          inference=True)

    elif opt.dataset_mode == 'kaist':
        from patagona_common.data.datasets import KAISTTemporalDataset
        dataset = KAISTTemporalDataset(root_dir='/home/ubuntu/datasets',
                                       video_sets=['set00', 'set01', 'set02', 'set03', 'set04', 'set05'])
    elif opt.dataset_mode == 'kaist_test':
        from patagona_common.data.datasets import KAISTTemporalDataset
        dataset = KAISTTemporalDataset(root_dir='/home/ubuntu/datasets',
                                       video_sets=['set06', 'set07', 'set08', 'set09', 'set10', 'set11'],
                                       random_crop=False,
                                       n_seq_frames=opt.n_frames_G,
                                       output_dim=(opt.loadSize, opt.loadSize),
                                       start_frame=opt.start_frame)
    elif opt.dataset_mode == 'kaist_test_single':
        from patagona_common.data.datasets import KAISTTemporalDataset
        dataset = KAISTTemporalDataset(root_dir='/home/ubuntu/datasets',
                                       video_sets=['set06', 'set07', 'set08', 'set09', 'set10', 'set11'],
                                       random_crop=False,
                                       n_seq_frames=opt.n_frames_G,
                                       output_dim=(opt.loadSize, opt.loadSize),
                                       all_first=True,
                                       start_frame=opt.start_frame)

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
