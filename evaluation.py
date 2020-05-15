from models import SCL_model,Descriptor_net

import numpy as np
import cv2
import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('GPS is available: ',torch.cuda.is_available())

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class SCL_evaluate:
    def __init__(self,width=256,height=256):
        self.writer=SummaryWriter('evals/SCL_2')
        self.spatial_features_size=20
        self.ref_length=50
        self.width=width
        self.height=height

        # create and load SCL model
        self.load_from="models/SCL.pth"
        self.model=SCL_model(self.spatial_features_size,width,height).to(device)
        self.model.load_state_dict(torch.load(self.load_from, map_location=device))
        self.model.eval()

        # creat and load descriptor model
        self.d_load_from="models/d_SCL.pth"
        self.d_model=Descriptor_net(self.spatial_features_size,width,height).to(device)
        self.d_model.load_state_dict(torch.load(self.d_load_from,map_location=device))
        self.d_model.eval()

        # I/O transforms
        mean=(0.485,0.450,0.406)
        std=(0.229,0.224,0.225)
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
        self.unnormalize=UnNormalize(mean,std)

        # read eval_videos directory
        self.path = "videos"
        filenames = [p for p in os.listdir(self.path) if p[0] != '.']
        self.video_paths = [os.path.join(self.path, f) for f in filenames]
        self.video_count = len(self.video_paths)
        print("The number of the videos: ", self.video_count)
        print("Videos in eval_videos directory: ")
        for i in range(self.video_count):
            print("%d. %s" % (i, self.video_paths[i]))

    def read_video(self,path):
        frames=np.empty((self.ref_length,self.height,self.width,3),dtype=np.uint8)
        cap=cv2.VideoCapture(path)
        length=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        factor=self.ref_length/length
        for i in range(length):
            ret,frame=cap.read()
            if ret:
                frame=cv2.resize(frame,(self.width,self.height))
                frame=cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
                index=int(i * factor)
                frames[index,...]=frame
        cap.release()
        return frames

    def eval(self):
        x = input("\n Please a video select for evaluation (by typing its number)")
        x = int(x)
        if x > self.video_count:
            raise AssertionError("Choose a number between 0 and %d" % self.video_count)
        frames = self.read_video(self.video_paths[x])
        embeddings=torch.empty(len(frames),32)
        # run evaluation
        for i in range(len(frames)):
            img=self.transform(frames[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                embeddings[i],spatial_features=self.model(img)
                vis2=self.d_model(spatial_features)
            vis1=self.unnormalize(img.squeeze()).to(device)
            vis2=self.unnormalize(vis2.squeeze())
            self.writer.add_image("descriptor prediction",torchvision.utils.make_grid([vis1,vis2]),i)
        # plot the rewards
        for  i in range(len(embeddings)):
            dis=torch.abs(embeddings[-1]-embeddings[i]).pow(2).sum(-1)
            cost=-dis
            self.writer.add_scalar("cost", cost, i)

a=SCL_evaluate()
a.eval()