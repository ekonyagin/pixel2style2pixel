#!/usr/bin/python
# encoding: utf-8
import os

import cv2
from torch.utils.data import Dataset


class GTResDataset(Dataset):

    def __init__(self, root_path, gt_dir=None, transform=None, transform_train=None):
        self.pairs = []
        for f in os.listdir(root_path):
            image_path = os.path.join(root_path, f)
            gt_path = os.path.join(gt_dir, f)
            if f.endswith(".jpg") or f.endswith(".png"):
                self.pairs.append([image_path, gt_path.replace(".png", ".jpg"), None])
        self.transform = transform
        self.transform_train = transform_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        from_path, to_path, _ = self.pairs[index]
        from_im = cv2.imread(from_path)
        from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)

        to_im = cv2.imread(to_path)
        to_im = cv2.cvtColor(to_im, cv2.COLOR_BGR2RGB)

        if self.transform:
            to_im = self.transform(to_im)
            from_im = self.transform(from_im)

        return from_im, to_im
