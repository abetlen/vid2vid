from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=300, help='how many test images to run')        
        self.parser.add_argument('--use_real_img', action='store_true', help='use real image for first frame')
        self.parser.add_argument('--start_frame', type=int, default=0, help='frame index to start inference on')        
        self.parser.add_argument('--n_frames_total', type=int, default=30, help='the overall number of frames in a sequence to test with')
        self.parser.add_argument('--test_video_scene', type=str, default='deathCircle', help='scene of test video from original dataset')
        self.parser.add_argument('--test_video_id', type=int, default=3, help='video id of test video from original dataset')
        self.isTrain = False
