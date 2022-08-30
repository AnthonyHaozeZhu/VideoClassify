# -*- coding: UTF-8 -*-
"""
@Project ：Classified 
@File ：data.py
@Author ：AnthonyZ
@Date ：2022/8/24 22:11
"""

from torch.utils.data.dataset import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import glob
from PIL import Image
import torch
from utils import *


class FramesLoader(Dataset):
    def __init__(self, pic_path, transform, opt, is_train=True):
        self.opt = opt
        all_vids, all_labels, self.catgs = get_vids(pic_path)
        self.labels_dict = {}
        self.transform = transform
        index = 0
        for i in self.catgs:
            self.labels_dict[i] = index
            index += 1
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
        train_indx, test_indx = next(sss.split(all_vids, all_labels))
        if is_train:
            self.vids = [all_vids[ind] for ind in train_indx]
            self.labels = [all_labels[ind] for ind in train_indx]
        else:
            self.vids = [all_vids[ind] for ind in test_indx]
            self.labels = [all_labels[ind] for ind in test_indx]

    def get_catgs(self):
        return self.catgs

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, item):
        imgs_path = glob.glob(self.vids[item] + "/*.jpg")
        label = self.labels_dict[self.labels[item]]
        frames = []
        for p2i in imgs_path:
            frame = Image.open(p2i)
            frames.append(frame)
        frames_tr = []
        for frame in frames:
            frame = self.transform(frame)
            frames_tr.append(frame)
        if frames_tr is None or len(frames_tr) == 0:
            frames_tr = []
            for i in range(16):
                frames_tr.append(torch.Tensor(3, self.opt.frame_height, self.opt.frame_width))
                label = self.labels_dict[self.labels[item]]
        return torch.stack(frames_tr[0: self.opt.n_frames]), torch.tensor(label)
