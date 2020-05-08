from models import SCL_model,Descriptor_net
from losses import TripletLoss,PixelwiseLoss, PixelTripletLoss
from dataset import SCL_dataset
import cv2

from tqdm import trange,tqdm

import torch
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("GPU is working:",torch.cuda.is_available())

class SCL_trainer:
    def __init__(self,load=True, width=256, height=256):
        self.writer=SummaryWriter('runs/SCL_1')
        self.spatial_features_size=20
        self.load=load
        self.dataset = SCL_dataset(width,height)
        self.model=SCL_model(self.spatial_features_size,width,height).to(device)
        self.descriptor_model=Descriptor_net(self.spatial_features_size,width,height).to(device)
        self.contrastive_criterion=TripletLoss()
        # self.pixel_criterion=PixelwiseLoss(width, height)
        self.pixel_criterion=PixelTripletLoss(width,height)
        self.load_from = "models/SCL.pth"
        self.save_to = "models/SCL.pth"
        self.d_load_from = "models/d_SCL.pth"
        self.d_save_to = "models/d_SCL.pth"
        if self.load:
            self.model.load_state_dict(torch.load(self.load_from, map_location=device))
            self.model.eval()
            self.delayed_model = SCL_model(self.spatial_features_size, width, height).to(device)
            self.delayed_model.load_state_dict(torch.load(self.load_from, map_location=device))
            self.delayed_model.eval()
            self.descriptor_model.load_state_dict(torch.load(self.d_load_from,map_location=device))
            self.descriptor_model.eval()
            self.d_delayed_model = Descriptor_net(self.spatial_features_size,width,height).to(device)
            self.d_delayed_model.load_state_dict(torch.load(self.d_load_from, map_location=device))
            self.d_delayed_model.eval()
            self.tau = 0.001
        self.optimizer=optim.Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-4)
        self.descriptor_optimizer=optim.Adam(self.descriptor_model.parameters(),lr=1e-4,weight_decay=1e-4)
        self.max_iter=3500
        self.dataloader = torch.utils.data.DataLoader(self.dataset, 1, shuffle=True, pin_memory=device)
        self.descriptor_dataloader = torch.utils.data.DataLoader(self.dataset, 1, shuffle=True, pin_memory=device)
        self.images = self.dataset.frames[0]
        self.embedding=torch.empty((len(self.images),32))
        self.cost_counter=0

    def run_training(self,iterations=200):
        self.dataset.count=0
        i=0
        for epoch in tqdm(range(iterations)):
            for data in self.dataloader:
                step=i
                # inputs
                data=data.permute(1,0,2,3,4).to(device)
                anchor,pos,neg=data
                # outputs
                anchor_embedding,spatial_features_anchor=self.model(anchor)
                pos_embedding,spatial_features_pos=self.model(pos)
                neg_embedding,spatial_features_neg=self.model(neg)
                loss = self.contrastive_criterion(anchor_embedding, pos_embedding, neg_embedding)
                # optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # plot loss
                # normalized_loss=loss.item()/self.contrastive_criterion.margins[self.contrastive_criterion.margin_step]
                self.writer.add_scalar("training_loss", loss.item(), step)
                # update margin - step the scheduler
                if (i+1)%3500==0:
                    self.dataset.update_margins()
                    self.contrastive_criterion.update_margin()
                if (i+1)%3500==0 or i == 0:
                    for j in range(len(self.images)):
                        img = self.dataset.transform(self.images[j]).unsqueeze(0).to(device)
                        with torch.no_grad():
                            self.embedding[j], _ = self.model(img)
                    for j in range(len(self.images)):
                        dis=torch.abs(self.embedding[-1]-self.embedding[j]).pow(2).sum(-1)
                        cost=-dis
                        self.writer.add_scalar("cost_%d" %self.cost_counter, cost, j)
                    self.writer.add_embedding(self.embedding, global_step=step)
                    self.cost_counter += 1
                i += 1
            # soft update
            if self.load:
                for param1, param2 in zip(self.model.parameters(), self.delayed_model.parameters()):
                    param2.data.copy_(self.tau * param1.data + (1 - self.tau) * param2.data)
                    param1.data.copy_(param2)
            # save the model and visualize the embeddings
            torch.save(self.model.state_dict(), self.save_to)

    def train_descriptors(self,iterations=200):
        self.dataset.count = 0
        i = 0
        for epoch in tqdm(range(iterations)):
            for data in self.descriptor_dataloader:
                step = i
                # inputs
                data = data.permute(1, 0, 2, 3, 4).to(device)
                img1, img2, _ = data
                # outputs
                embedding1, spatial_features1 = self.model(img1)
                embedding2, spatial_features2 = self.model(img2)
                # to be used in pixel loss
                img1_out=self.descriptor_model(spatial_features1)
                vis1 = self.dataset.unormalize(img1.squeeze())
                vis2 = self.dataset.unormalize(img1_out.squeeze())
                img1_un=vis1.permute(1,2,0)
                img2_out = self.descriptor_model(spatial_features2)
                img2_un = self.dataset.unormalize(img2.squeeze()).permute(1,2,0)
                if (i+1)%3500==0:
                    self.writer.add_image("original vs. prediction", torchvision.utils.make_grid([vis1,vis2]),step)
                    # if i>10000:
                    self.dataset.update_margins()
                    self.pixel_criterion.update_margins()
                # loss computation
                loss_p = self.pixel_criterion(img1_un,img1_out,img2_un,img2_out)
                # optimization
                self.descriptor_optimizer.zero_grad()
                loss_p.backward()
                self.descriptor_optimizer.step()
                # plot loss
                # normalized_loss=loss_p.item()/self.pixel_criterion.margins[self.pixel_criterion.margin_step]
                self.writer.add_scalar("training_descriptor_loss",loss_p.item(),step)
                torch.save(self.descriptor_model.state_dict(), self.d_save_to)
                i += 1
            # soft update
            if self.load:
                for param1, param2 in zip(self.descriptor_model.parameters(), self.d_delayed_model.parameters()):
                    param2.data.copy_(self.tau * param1.data + (1 - self.tau) * param2.data)
                    param1.data.copy_(param2)

scl=SCL_trainer(load=False)
scl.run_training()
scl.train_descriptors()