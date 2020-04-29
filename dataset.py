import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class SCL_dataset(Dataset):
    def __init__(self, width, height):
        self.margin = 5

        self.pos_margin = 3
        self.neg_margin_near = 5
        self.neg_margin_far = 10

        self.ref_length = 100
        # we will resize frames to this width and height
        self.width = width
        self.height = height
        self.num_channels = 3

        # read video_multiview directory
        self.path = "/home/ali/Cost-learning/videos"
        filenames = [p for p in os.listdir(self.path) if p[0] != '.']
        filenames = np.sort(filenames)
        self.video_paths = [os.path.join(self.path, f) for f in filenames]
        self.video_count = len(self.video_paths)
        # logging
        print("The number fo the videos:", self.video_count)
        print(" video paths:")
        for i in range(self.video_count):
            print("%d. %s" % (i, self.video_paths[i]))

        # collect frames
        self.frames = self.read_videos()

        # transform - normalization needed for inception pretrained model
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            # transforms.ColorJitter(0.0,0.0,0.0,0.0)
        ])
        self.unormalize = UnNormalize(mean, std)

    def __len__(self):
        return len(self.frames)

    def update_margins(self,pos_factor, neg_near_factor, neg_far_factor):
        self.pos_margin*=pos_factor
        self.neg_margin_near*=neg_near_factor
        self.neg_margin_far*=neg_far_factor

    def __getitem__(self, idx):
        pos_index = self.sample_positive(idx)
        neg_index = self.sample_negative(idx)
        anchor = self.transform(self.frames[idx])
        pos = self.transform(self.frames[pos_index])
        neg = self.transform(self.frames[neg_index])
        sample = torch.stack([anchor, pos, neg])
        return sample

    def show_sample(self, idx):
        sample = self.__getitem__(idx)
        anchor = sample[0].permute(1, 2, 0)
        pos = sample[1].permute(1, 2, 0)
        neg = sample[2].permute(1, 2, 0)
        image = np.hstack([self.unormalize(anchor), self.unormalize(pos), self.unormalize(neg)])
        cv2.imshow("anchor, pos, neg", image)
        cv2.waitKey(0)

    def read_videos(self):
        frames = []
        for p in range(self.video_count):
            videp_path = self.video_paths[p]
            cap = cv2.VideoCapture(videp_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            factor = self.ref_length / length
            frames = np.empty((length, self.height, self.width, self.num_channels), dtype=np.uint8)
            for i in range(length):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (self.width, self.height))
                    index = int(i * factor)
                    frames[index, ...] = frame
            cap.release()
            frames.append(frames)
        frames1 = np.array(frames)#.squeeze(0)
        return frames1

    def sample_positive(self, anchor):
        min_ = max(0, anchor - self.pos_margin)
        max_ = min(self.ref_length - 1, anchor + self.pos_margin)
        return np.random.choice(np.arange(min_, max_))

    def sample_negative(self, anchor):
        start1 = max(0, anchor - self.neg_margin_far)
        end1 = max(0, anchor - self.neg_margin_near)
        range1 = np.arange(start1, end1)
        start2 = min(self.ref_length-1, anchor + self.neg_margin_near)
        end2= min(self.ref_length-1, anchor + self.neg_margin_far)
        range2 = np.arange(start2, end2)
        range = np.concatenate([range1, range2])
        return np.random.choice(range)