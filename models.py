import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SCL_model(nn.Module):
    def __init__(self,descriptor_dims ,width ,height):
        super().__init__()
        # comment lines 106-117 from ./local/lib/python3.6/site-packages/torchvision/models/inception.py
        # to make it load faster
        # those line initialize weights, they slow down the code because of a problem with scipy.stats
        # we are using pretrained inception model, so we don't need that
        self.inception = models.inception_v3(pretrained=True, progress=True)
        self.conv1 = self.inception.Conv2d_1a_3x3
        self.conv2_1 = self.inception.Conv2d_2a_3x3
        self.conv2_2 = self.inception.Conv2d_2b_3x3
        self.conv3 = self.inception.Conv2d_3b_1x1
        self.conv4 = self.inception.Conv2d_4a_3x3
        self.mix5_1 = self.inception.Mixed_5b
        self.mix5_2 = self.inception.Mixed_5c
        self.mix5_3 = self.inception.Mixed_5d
        # freezing inception model
        for param in self.parameters():
            param.requires_grad=False
        self.conv6_1 = nn.Conv2d(288, 100, kernel_size=3, stride=1)
        self.batch_norm_1 = nn.BatchNorm2d(100, eps=1e-3)
        self.conv6_2 = nn.Conv2d(100, 20, kernel_size=3, stride=1)
        self.batch_norm_2 = nn.BatchNorm2d(20, eps=1e-3)
        # softmax2d is an activation in each channel
        # spatial softmax s_{ij}=\frac{exp(a_{ij})}{\sum_{i',j'} exp(a_{i'j'})}
        self.spatial_softmax = nn.Softmax2d()
        self.fc7 = nn.Linear(20 * (width/8-7) * (height/8-7) , 32)
        self.alpha = 10.0

    def normalize(self, x):
        norm_const = torch.pow(x, 2).sum(1).add(1e-10)
        norm_const = norm_const.sqrt()
        output = torch.div(x, norm_const.view(-1, 1).expand_as(x))
        return output

    def forward(self, x):
        input_dims = x.size()[2:]
        # 3 x 160 x 320 (C x W x H)
        x = self.conv1(x)
        # 32 x 79 x 159 (32 x W/2-1 x H/2-1)
        x = self.conv2_1(x)
        # 32 x 77 x 157 (32 x W/2-3 x H/2-3)
        x = self.conv2_2(x)
        # 64 x 77 x 157 (64 x W/2-3 x H/2-3)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 64 x 38 x 78 (64 x W/4-2 x H/4-2)
        x = self.conv3(x)
        # 80 x 38 x 78 (80 x W/4-2 x H/4-2)
        x = self.conv4(x)
        # 192 x 36 x 76  (192 x W/4-4 x H/4-4)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 192 x 17 x 37 (192 x W/8-3 x H/8-3)
        x = self.mix5_1(x)
        # 256 x 17 x 37 (256 x W/8-3 x H/8-3)
        x = self.mix5_2(x)
        # 288 x 17 x 37 (288 x W/8-3 x H/8-3)
        x = self.mix5_3(x)
        # 288 x 17 x 37 (256 x W/8-3 x H/8-3)
        x = self.conv6_1(x)
        # 100 x 15 x 35 (100 x W/8-5 x H/8-5)
        x = self.batch_norm_1(x)
        # 100 x 15 x 35 (100 x W/8-5 x H/8-5)
        x = self.conv6_2(x)
        # 20 x 13 x 33 (20 x W/8-7 x H/8-7)
        x = self.batch_norm_2(x)
        # 20 x 13 x 33 (20 x W/8-7 x H/8-7)
        x = self.spatial_softmax(x)
        y = self.fc7(x.view(x.size()[0], -1))
        # outputs the embedding and the spatial features
        return self.normalize(y) * self.alpha, x

class Descriptor_net(nn.Module):
    def __init__(self, spatial_features_size, width,height):
        super().__init__()
        self.conv_trans1=nn.ConvTranspose2d(spatial_features_size,6,kernel_size=3,stride=1)
        self.conv_trans1 = nn.ConvTranspose2d(6, 3, kernel_size=3, stride=1)
        self.width=width
        self.height=height

    def forward(self,x):
        x=self.conv_trans1(x)
        x=F.relu(x)
        x=self.conv_trans2(x)
        x=F.relu(x)
        x=F.interpolate(x,size=(self.width,self.height),mode='bilinear',align_corners=True)
        return x