import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import math
from pathlib import Path
import sys


class VideoReader():
    def __init__(self, min_frames=10, channels=3, width=1280, height=720):
        self.min_frames = min_frames
        self.channels = channels
        self.width = width
        self.height = height

    def _read_video(self, video_name):
        # Open the video file
        cap = cv2.VideoCapture(Path(video_name).as_posix())

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert self.min_frames <= num_frames, AssertionError(f'Video {video_file}, minimum frames condition unsatisfied')
        # assert self.width == width, AssertionError(f'Video {video_file}, video has different width = {width}')
        # assert self.height == height, AssertionError(f'Video {video_file}, video has different height = {height}')

        frames = torch.FloatTensor(num_frames, self.channels, self.height, self.width)
        
        for f in range(num_frames):
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame, (self.width, self.height))
                frame = torch.from_numpy(frame)
                frame = frame.permute(2, 0, 1) # HWC2CHW
                frames[f, :, :, :] = frame
            else:
                raise AssertionError(f'Could Not read all the frames in the video {video_file}')
    
        return frames / 255.0 #TODO Add correct transforms

    def generate_slices(self, video_name, slice_length=10, stride=5):
        # reading the video
        try:
            video = self._read_video(video_name)
        except AssertionError as e:
            print(e)
            return None
        except Exception as e:
            # print(f'Unexpected error while reading video {video_name}: ', sys.exc_info()[0])
            print(f'Unexpected error while reading video {video_name}: ', e)
            return None

        num_frames = len(video)
        num_slices = math.ceil(num_frames/stride) # maximum number of slices

        # generating slices with the given stride
        slices = [video[i:i+slice_length] for i in range(num_slices) if (i+slice_length) < num_frames]
        return torch.stack(slices, axis=0).permute(0, 2, 1, 3, 4)


class VideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(
        self,
        root_dir,
        channels=3,
        width=1280,
        height=720,
        slice_length=10,
        stride=5,
        min_frames=10
    ):
        """
        Args:
            root_dir (string): Directory with all the videos. Each class must be in a different folder
            channels: Number of channels of frames
            width, height: Dimensions of the frames
            slice_length: Number of frames to be loaded in a video slice
            stride: step size when generating video slices
            min_frames: parameter to ignore videos with fewer frames than min_frames
        """

        self.root_dir = root_dir
        self.channels = channels
        self.width = width
        self.height = height
        self.slice_length = slice_length
        self.stride = stride
        self.min_frames = min_frames

        self.video_reader = VideoReader(min_frames, channels, width, height)

        self.class_dirs = [p.as_posix() for p in Path(root_dir).glob('*') if p.is_dir()]
        assert len(self.class_dirs) > 0, AssertionError('root_dir must contain folders of different classes')

        classes = sorted([c.split('/')[-1] for c in self.class_dirs])
        self.classes = {c:i for i,c in enumerate(classes)}

        # self.video_names = itertools.chain(*[list(Path(p).glob('*.mp4')) for p in self.class_dirs])
        self.video_names = []
        for class_name in self.class_dirs:
            video_in_class = list(Path(class_name).glob('*.mp4'))
            self.video_names += video_in_class
            # print(f'Found {len(video_in_class)} videos in class {class_name.split('/')[-1]}')
            print(f"Found {len(video_in_class)} videos in class {class_name.split('/')[-1]}")
        
        print(f'Total videos found: {len(self.video_names)}')

    def get_label(self, class_name):
        # return name of parent folder
        return self.classes.get(class_name)

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):

        video_name = self.video_names[idx].as_posix()
        class_name = video_name.split('/')[-2]
        label = self.get_label(class_name)

        slices = self.video_reader.generate_slices(video_name, self.slice_length, self.stride)
        if slices != None: return slices, [label]*len(slices)

    def collate(self, batch):
        ret_slices, ret_labels = [], []        
        for slc, lbl in filter(None, batch):
            ret_slices += slc
            ret_labels += lbl
        return (torch.stack(ret_slices, axis=0), torch.LongTensor(ret_labels)) if len(ret_labels) > 0 else None

# [[rgb, rgb, rgb], [rgb, rgb, rgb], [rgb, rgb, rgb], [rgb, rgb, rgb]]
# (batch_size, slice_length, C, W, H)