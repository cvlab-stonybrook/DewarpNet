####
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy import misc
from scipy import io
np.set_printoptions(threshold=np.nan)
from math import exp
import cv2
import os
import matplotlib.pyplot as plt


class JointLoss(torch.nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()

    def forward(self, pred,label):
        (batch_size, channel, h, w) = pred.size()

        batch_mask=(label!=0).float()
        fg_pred=pred*batch_mask
        bg_pred=pred*(1-batch_mask)
        bg_label=label*(1-batch_mask)

        # get BCE loss for the background
        # bg_loss_fn = nn.BCELoss() # BCE doesn't work


        # get L1 loss for the foreground
        fg_loss_fn = nn.L1Loss()

        # get hinge loss for background 
        bg_loss=torch.mean(torch.max(torch.max(torch.max(bg_pred,bg_label),dim=-1)[0],dim=-1)[0])
        fg_loss=fg_loss_fn(fg_pred,label)

        return fg_loss,bg_loss



# def check_mask (inp_images,inp_labels):
#   batch_mask=torch.zeros(inp_labels.shape)
#   fg_images=torch.zeros(inp_images.shape)
#   batch_mask=(inp_labels!=0).float()
#   # for i in range(9):
#   #   print (inp_labels[i].shape)
#   #   batch_mask[i,0,:,:]=batch_mask[i,1,:,:]=batch_mask[i,2,:,:]=((inp_labels[i,0,:,:]!=0)&(inp_labels[i,1,:,:]!=0)&(inp_labels[i,2,:,:]!=0)).float()
#   #   fg_images[i,:,:,:]=inp_images[i,:,:,:]*batch_mask[i,:,:,:]
#   #   plt.imshow(inp_labels[i].numpy().transpose(1,2,0))
#   #   # plt.imshow(inp_images[i].numpy().transpose(1,2,0))
#   #   plt.show()

#   fg_images=inp_images*batch_mask
#   bg_images=inp_images*(1-batch_mask)
#   f,axarr=plt.subplots(9,5)
#   # l1 on fg images
#   # bce on batch_mask,pred


#   fg_imgs=fg_images.numpy()
#   bg_imgs=bg_images.numpy()
#   msks=batch_mask.numpy()
#   for i in range(9):
#       axarr[i][0].imshow(fg_imgs[i].transpose(1,2,0))
#       axarr[i][1].imshow(bg_imgs[i].transpose(1,2,0))
#       axarr[i][2].imshow(msks[i,0,:,:])
#       axarr[i][3].imshow(msks[i,1,:,:])
#       axarr[i][4].imshow(msks[i,2,:,:])
#   plt.show()





# # read images from the dataset
# data_path='/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/'
# filenames=['2/34275Xpr_Page_319X224X0001','2/30548Xcp_Page_0085X273X0001','2/30111Xtc_Page_081X259X0001',
#            '2/31362Xns_Page_639X309X0001','2/33711Xcp_Page_0685X265X0001','2/31864Xns_Page_458X58X0001',
#            '2/32548Xns_Page_002X13X0001','2/34952Xny_Page_060X291X0001','2/32718Xny_Page_261X18X0001']
# # # For testing
# if __name__ == '__main__':


#     batch_img=[]
#     batch_wc=[]
#     for f in filenames:
#         #read file 
#         curr_img = misc.imread(data_path+'images-corrmesh/'+f+'.png')[:,:,:3]
#         batch_img.append(curr_img)

#       lbl_name = f.strip().split('X')
#         foldr,lbl_id=lbl_name[0].split('/')
#         lbl_name = 'WCX'+lbl_id+lbl_name[3]+'.exr'         #WCX20001.exr
#         lbl_path = os.path.join(data_path, 'wc-corrmesh',foldr,lbl_name)
#         # print(lbl_path)
#         lbl=cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#         # plt.imshow(lbl)
#         # plt.show()

#         batch_wc.append(lbl)

#     img_size=(284,284)
#     images=np.full((9,3,img_size[0],img_size[1]),0.0)
#     idx=0
#     for bimg in batch_img:
#         bimg = misc.imresize(bimg, (img_size[0], img_size[1]))
#         bimg = bimg[:, :, ::-1]
#         bimg = bimg.astype(np.float64)
        
#         bimg = bimg.astype(float) / 255.0
#         # NHWC -> NCHW
#         bimg = bimg.transpose(2, 0, 1)
#         # print (bimg.shape)
#         images[idx,:,:,:]=bimg
#         idx+=1

#     labels=np.full((9,3,img_size[0],img_size[1]),0.0)
#     idx=0
#     for bwc in batch_wc:
#       bwc = bwc.astype(float)
#         #normalize label
#         msk=((bwc[:,:,0]!=0)&(bwc[:,:,1]!=0)&(bwc[:,:,2]!=0)).astype(np.uint8)*255

#         ####################################### Uncomment when not just mask
#         xmx, xmn, ymx, ymn,zmx, zmn=5.672155, -5.657737, 5.984079, -5.8917637, 1.599145, -3.443543
        
#         bwc[:,:,0]= (bwc[:,:,0]-zmn)/(zmx-zmn)
#         bwc[:,:,1]= (bwc[:,:,1]-ymn)/(ymx-ymn)
#         bwc[:,:,2]= (bwc[:,:,2]-xmn)/(xmx-xmn)
#         bwc=cv2.bitwise_and(bwc,bwc,mask=msk)
#         ######################################
#         bwc = cv2.resize(bwc, (img_size[0], img_size[1]), cv2.INTER_NEAREST)
#         # plt.imshow(bwc)
#         # plt.show()
#         bwc.astype(float)
#         # NHWC -> NCHW
#         bwc = bwc.transpose(2, 0, 1)
#         labels[idx,:,:,:]=bwc
#         idx+=1
    
#     print (images.shape)
#     inp_images = torch.from_numpy(images).float()
#     inp_labels = torch.from_numpy(labels).float()

#     check_mask (inp_images,inp_labels)



