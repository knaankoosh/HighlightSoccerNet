import logging
import random

import numpy as np
import os
import torch
import skimage.transform
from moviepy.editor import VideoFileClip
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    """
    A Container for Highlight and non highlight clips.
    Assumes a path to a text file containing paths to the clips.
    i.e "filelist_train.txt", if one doesnt exist use 'OrganizeData' function.
    Every highlight clip should start with HL and non highlight should start with NOHL
    """

    def __init__(self, path, transform=False):
        with open(path, 'r') as f:
            self.clips = f.read().splitlines()

        self.transform = None
        if transform:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        with VideoFileClip(self.clips[idx]) as clip:
            # Audio stream
            cut = lambda i: clip.audio.subclip(i, i + 1).to_soundarray(fps=22000)
            volume = lambda array: np.sqrt(((1.0 * array) ** 2).mean())
            volumes = [volume(cut(i)) for i in range(0, int(clip.duration - 1))]
            volumes = np.array(volumes, dtype='float32')
            volumes = np.append(volumes, [volumes[-1], volumes[-1]])

            # Video stream
            frames = np.zeros((int(clip.duration) + 1, 224, 224, 3))
            for i, frame in enumerate(clip.iter_frames(fps=1)):
                    frames[i] = skimage.transform.resize(frame, (224,224))
            frames = frames.transpose([0, 3, 1, 2]).astype('float32')

            label = np.float32(1) if os.path.basename(self.clips[idx])[:2] == "HL" else np.float32(0)

        return frames, volumes, label



def initialize_loaders(args):
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        VideoDataset(args.train_data, transform=True),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        VideoDataset(args.test_data, transform=False),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    return train_loader, test_loader

def organize_data(path):
    """
    Separates data to test and train sets.
    Creates the filelists text files.
    Assumes path contains:
      - a subfolder named "HL"
      - a subfolder named "NO_HL"
    """

    path = os.path.abspath(path)

    # Read clip names
    hl_clips = os.listdir(os.path.join(path, 'HL'))
    nohl_clips = os.listdir(os.path.join(path, 'NO_HL'))

    test_input = []
    train_input = []

    for f in hl_clips:
        if np.random.rand() < 0.1:
            test_input.append(os.path.join(path, 'HL', f))
        else:
            train_input.append(os.path.join(path, 'HL', f))

    for f in nohl_clips:
        if np.random.rand() < 0.1:
            test_input.append(os.path.join(path, 'NO_HL', f))
        else:
            train_input.append(os.path.join(path, 'NO_HL', f))

    with open(os.path.join(path,'filelist_train.txt'), 'w+') as fid:
        print(fid)
        for f in train_input:
            print(f, file=fid)
    with open(os.path.join(path,'filelist_test.txt'), 'w+') as fid:
        print(fid)
        for f in test_input:
            print(f, file=fid)