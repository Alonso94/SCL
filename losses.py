import numpy as np
import random
import cv2
import itertools
import math

import torch
import torch.nn as nn

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("GPU is working:",torch.cuda.is_available())

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin_step=0
        self.count=0
        self.margins=[3,4,5,6]

    def update_margin(self):
        self.count+=1
        self.margin_step = self.count % len(self.margins)

    def distance(self, x, y):
        diff = torch.abs(x - y)
        diff=torch.pow(diff, 2).sum(-1)
        return diff

    def forward(self,anchor,pos,neg):
        pos_distance=self.distance(anchor,pos)
        neg_distance=self.distance(anchor,neg)
        loss = torch.clamp( self.margins[self.margin_step] + pos_distance - neg_distance,min=0.0).mean()
        return loss

class PixelTripletLoss(nn.Module):
    def __init__(self,width, height):
        super().__init__()
        self.margin_step = 0
        self.count = 0
        self.margins = [5, 10, 20, 30]
        self.num_pos_points = 500
        self.num_neg_points = 500

        self.width = width
        self.height = height
        self.num_channels = 3

    def update_margins(self):
        self.count += 1
        self.margin_step = self.count % len(self.margins)

    def edge_detection(self,img):
        img=img.cpu().numpy()*255
        img=img.astype(np.uint8)
        edges=cv2.Canny(img,100,220)
        indices=np.where(edges!=[0])
        coordinates=list(zip(indices[1],indices[0]))
        if len(coordinates)<self.num_pos_points:
            a=[range(0,255)]*2
            possible_points=list(itertools.product(*a))
            coordinates+=random.sample(possible_points,self.num_pos_points-len(coordinates))
        else:
            coordinates=random.sample(coordinates,self.num_pos_points)
        return torch.from_numpy(np.array(coordinates)).to(device)

    def distance(self, x, y):
        diff = torch.abs(x - y)
        diff = torch.pow(diff, 2).sum(-1)
        epsilon=1e-7
        return (diff+epsilon).sqrt()

    def features_in_out(self,img,features):
        xs=features[:,0]
        ys=features[:,1]
        return img[:,:,xs,ys]

    def loss_matches(self,img1_out,img2_out,features_in_out1,features_in_out2):
        loss=self.distance(features_in_out1,features_in_out2).mean()
        return loss

    def loss_non_matches(self,img1_out,img2_out,features_in_out1):
        non_mathces_x=torch.randint(0,self.width,(1,self.num_neg_points)).long()
        non_mathces_y = torch.randint(0, self.height, (1, self.num_neg_points)).long()
        non_matches=torch.stack([non_mathces_x,non_mathces_y],dim=1)
        features_in_out2=self.features_in_out(img2_out,non_matches)
        loss=self.distance(features_in_out1,features_in_out2)
        return loss

    def forward(self,img1,img1_out,img2,img2_out):
        # extract features (indices)
        features_img1=self.edge_detection(img1)
        features_img2=self.edge_detection(img2)
        # features from the out image
        features_in_out1 = self.features_in_out(img1_out, features_img1)
        features_in_out2 = self.features_in_out(img2_out, features_img2)
        # matches loss
        pos_distance=self.loss_matches(img1_out,img2_out,features_in_out1,features_in_out2)
        # non_matches loss
        neg_distance=self.loss_non_matches(img1_out,img2_out,features_in_out1)
        loss = torch.clamp(self.margins[self.margin_step] + pos_distance - neg_distance, min=0.0).mean()
        return loss


class PixelwiseLoss(nn.Module):
    def __init__(self,width, height):
        super(PixelwiseLoss, self).__init__()
        self.margin=0.5
        self.num_pos_points=300
        self.num_neg_points=300

        self.width=width
        self.height=height
        self.num_channels=3

    def edge_detection(self,img):
        img=img.cpu().numpy()*255
        img=img.astype(np.uint8)
        edges=cv2.Canny(img,100,220)
        # cv2.imshow("img", img)
        # cv2.imshow("edges", edges)
        # cv2.waitKey(0)
        indices=np.where(edges!=[0])
        coordinates=list(zip(indices[1],indices[0]))
        # print(len(coordinates))
        if len(coordinates)<self.num_pos_points:
            a=[range(0,255)]*2
            possible_points=list(itertools.product(*a))
            coordinates+=random.sample(possible_points,self.num_pos_points-len(coordinates))
        else:
            coordinates=random.sample(coordinates,self.num_pos_points)
        return torch.from_numpy(np.array(coordinates)).to(device)

    def distance(self, x, y):
        diff = torch.abs(x - y)
        diff = torch.pow(diff, 2).sum(-1)
        epsilon=1e-7
        return (diff+epsilon).sqrt()

    def features_in_out(self,img,features):
        xs=features[:,0]
        ys=features[:,1]
        return img[:,:,xs,ys]

    def ring_around(self,features):
        # ring with width 25 and inner radius 5 pixels
        repeat=features.repeat(self.num_neg_points,1)
        num_offsets=repeat.size()[0]
        radius=torch.rand((1,num_offsets))*25+5
        angle=torch.rand((1,num_offsets))*(2*math.pi)
        x_offsets=radius*torch.cos(angle)
        y_offsets=radius*torch.sin(angle)
        x_offsets=x_offsets.to(device)
        y_offsets=y_offsets.to(device)
        nonmatch_x=repeat[:,0]+x_offsets
        nonmatch_y=repeat[:,1]+y_offsets
        nonmatch=torch.stack([nonmatch_x,nonmatch_y],dim=1).squeeze()
        # ensure points inside the image
        nonmatch=torch.clamp(nonmatch,0,self.height-1).long().transpose(1,0)
        # nonmatch.view(self.num_neg_points*self.num_pos_points,2)
        return nonmatch

    def feature_movements(self,img1,img2):
        img1 = img1.cpu().numpy() * 255
        img1 = img1.astype(np.uint8)
        edges1=cv2.Canny(img1,100,200)
        img2 = img2.cpu().numpy() * 255
        img2 = img2.astype(np.uint8)
        edges2=cv2.Canny(img2,100,200)
        edges_diff=edges1-edges2
        indices=np.where(edges_diff!=[0])
        coordinates = list(zip(indices[1], indices[0]))
        return torch.from_numpy(np.array(coordinates)).to(device)

    def loss_matches(self,img1_out,img2_out,features_img1,features_img2):
        features_in_out1=self.features_in_out(img1_out,features_img1)
        features_in_out2=self.features_in_out(img2_out,features_img2)
        loss=self.distance(features_in_out1,features_in_out2).mean()
        return loss

    def loss_non_matches(self, img1_out, img2_out, features_img1, features_img2):
        # non matches are a ring around the matches
        ring_around1=self.ring_around(features_img1)
        ring_around2=self.ring_around(features_img2)
        # print(ring_around2.shape,features_img1.shape)
        # we need the distance between features of img1 and ring around in img2
        # and vice-versa
        features_in_out1 = self.features_in_out(img1_out, features_img1)
        features_in_out2 = self.features_in_out(img2_out, ring_around2)
        features_in_out1=features_in_out1.repeat(1,1,self.num_neg_points)
        dis1 = self.distance(features_in_out1, features_in_out2)
        loss1=torch.add(torch.neg(dis1),self.margin)
        zeros=torch.zeros_like(loss1)
        loss1=torch.max(zeros,loss1).mean()
        features_in_out1 = self.features_in_out(img1_out, ring_around1)
        features_in_out2 = self.features_in_out(img2_out, features_img2)
        features_in_out2 = features_in_out2.repeat(1, 1, self.num_neg_points)
        dis2 = self.distance(features_in_out1, features_in_out2)
        loss2 = torch.add(torch.neg(dis2), self.margin)
        zeros = torch.zeros_like(loss2)
        loss2 = torch.max(zeros, loss2).mean()
        loss=loss1+loss2
        return loss

    def loss_diff(self, img1_out, img2_out, img1, img2):
        features_moved1=self.feature_movements(img1,img2)
        features_moved2 = self.feature_movements(img2, img1)
        features_in_out1=self.features_in_out(img1_out,features_moved1)
        features_in_out2=self.features_in_out(img2_out,features_moved2)
        loss=self.distance(features_in_out1,features_in_out2).mean()
        return loss

    def forward(self,img1,img1_out,img2,img2_out):
        # img1 and img3 (anchor and neg1) from the same videos
        # img2 and img4 (pos and neg2) from the same videos
        # img1 and img2 at the same time
        # img3 and img4 at the same time
        # extract features (edges)
        features_anchor=self.edge_detection(img1)
        features_pos=self.edge_detection(img2)
        # matches loss - images taken at the same time
        loss_matches=self.loss_matches(img1_out,img2_out,features_anchor,features_pos)
        # non_matches loss - non matches from images taken at the same time
        loss_non_matches=self.loss_non_matches(img1_out,img2_out,features_anchor,features_pos)
        # difference loss - difference between images from the same videos
        loss_difference=self.loss_diff(img1_out, img2_out,img1,img2)
        loss= loss_matches +  loss_non_matches #+ loss_difference
        return loss