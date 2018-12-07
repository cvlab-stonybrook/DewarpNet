#implementation of chamfer loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
import cv2
import matplotlib.pyplot as plt
import pytorch_ssim

# from ptsemseg.loader import get_loader, get_data_path

def unwarp(img, bm):
    
    # print(bm.type)
    # img=torch.from_numpy(img).cuda().double()
    img=img.double()
    # print(img.type)
    res = F.grid_sample(input=img, grid=bm)
    # print(res.shape)
    return res


class Unwarploss(torch.nn.Module):
    def __init__(self):
        super(Unwarploss, self).__init__()
        self.xmx, self.xmn, self.ymx, self.ymn = 166.28639310649825, -3.792634897181367, 189.04606710275974, -18.982843029373125

    def forward(self,inp,pred,label):
        #image [n,c,h,w], target_nhwc [n,h,w,c], labels [n,h,w,c]
        n,c,h,w=inp.shape           #this has 6 channels if image is passed
        # print (h,w)
        # inp=inp.detach().cpu().numpy()
        inp_img=inp[:,:3,:,:] #img in bgr 

        # denormalize pred
        pred=(pred/2.0)+0.5
        pred[:,:,:,0]=(pred[:,:,:,0]*(self.xmx-self.xmn)) +self.xmn
        pred[:,:,:,1]=(pred[:,:,:,1]*(self.ymx-self.ymn)) +self.ymn
        pred[:,:,:,0]=pred[:,:,:,0]/float(156.0)
        pred[:,:,:,1]=pred[:,:,:,1]/float(187.0)
        pred=(pred-0.5)*2
        pred=pred.double()

        # denormalize label
        label=(label/2.0)+0.5
        label[:,:,:,0]=(label[:,:,:,0]*(self.xmx-self.xmn)) +self.xmn
        label[:,:,:,1]=(label[:,:,:,1]*(self.ymx-self.ymn)) +self.ymn
        label[:,:,:,0]=label[:,:,:,0]/float(156.0)
        label[:,:,:,1]=label[:,:,:,1]/float(187.0)
        label=(label-0.5)*2
        label=label.double()

        uwpred=unwarp(inp_img,pred)
        uworg=unwarp(inp_img,label)
        loss_fn = nn.MSELoss()
        ssim_loss = pytorch_ssim.SSIM()
        uloss=loss_fn(uwpred,uworg)
        ssim = 1-ssim_loss(uwpred,uworg)

        # print(uloss)

        return uloss.float(),ssim.float(),uworg,uwpred

        

