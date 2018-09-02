import numpy as np # for numerical operations
from moviepy.editor import VideoFileClip
from data import VideoDataset

ds = VideoDataset(r'data/filelist_train.txt', 20, 256, 256)

vid = ds.__getitem__(1)
