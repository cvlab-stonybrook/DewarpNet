# Loader for end 2 end training 
# image, world coords, backward mapping, albedo image as input

import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
import cv2
import random
import hdf5storage as h5

from tqdm import tqdm
from torch.utils import data

from augmentations import call_augmentations


def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    js = open('../../config.json').read()
    data = json.loads(js)
    return os.path.expanduser(data[name]['data_path'])

class swat3de2eLoader(data.Dataset):
    """
    Data loader for the  semantic segmentation dataset.
    """
    def __init__(self, root, split='trainbmswat3d', is_transform=False,
                 img_size=512, bm_size=128, augmentations=None, img_norm=True):
        self.root = os.path.expanduser(root)
        self.altroot='/home/sagnik/DewarpNet/swat3d/'
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 2
        # self.mean = np.array([117.72199, 113.61581, 109.92113])
        self.mean = np.array([0.0, 0.0, 0.0])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
                                               else (img_size, img_size)
        self.bm_size = bm_size if isinstance(bm_size, tuple) \
                                               else (bm_size, bm_size)
        for split in ['trainbmswat3d', 'valbmswat3d']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        #self.setup_annotations()


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                 #1/2Xec_Page_453X56X0001.png
        im_path = pjoin(self.root, 'img',  im_name + '.png')  
        
        ################################################################### This is for Pre CVPR'18 data
        # lbl_name = im_name.strip().split('X')
        # foldr,lbl_id=lbl_name[0].split('/')
        # wc_name = 'WCX'+lbl_id+lbl_name[3]+'.exr'         #WCX20001.exr
        # wc_path = pjoin(self.root, 'wc-corrmesh',foldr,wc_name) 

        # bm_name = 'DCX'+lbl_id+lbl_name[3]+'.mat'         #DCX20001.exr
        # bm_path = pjoin(self.root, 'bm-corrmesh',foldr,bm_name) 

        # alb_name = 'ALXN'+lbl_id+lbl_name[3]+'.png'         #WCX20001.exr
        # alb_path = pjoin(self.root, 'alb-corrmesh',foldr,alb_name) 
        ################################################################### This is for Pre CVPR'18 data
        img_foldr,fname=im_name.split('/')

        if 'val' in self.split:
            recon_foldrs=['chess48']
        else:
            # recon_foldrs=['chess4','chess12','chess48']
            recon_foldrs=['chess48']

        choice=random.sample(range(len(recon_foldrs)), 1)
        recon_foldr=recon_foldrs[choice[0]]
        wc_path = pjoin(self.root, 'wc' , im_name + '.exr')
        if int(img_foldr) > 7:
            bm_path = pjoin(self.altroot, 'bm' , im_name + '.mat')
        else:
            bm_path = pjoin(self.root, 'bm' , im_name + '.mat')
        alb_path = pjoin(self.altroot,'recon',img_foldr,recon_foldr, fname[:-4]+recon_foldr+'0001.png')
        # print (alb_path)
        # alb_path = pjoin(self.root,'uv' , im_name + '.exr')
        # print(im_path)
        im = m.imread(im_path,mode='RGB')
        im = np.array(im, dtype=np.uint8)
        # img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)

        # # equalize the histogram of the Y channel
        # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # # convert the YUV image back to RGB format
        # im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        bm = h5.loadmat(bm_path)['bm']
        # alb = cv2.imread(alb_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        alb = m.imread(alb_path,mode='RGB')
        wc= np.array(wc, dtype=np.float)
        if self.augmentations:
            im, wc = call_augmentations(self.root,im_name,im, wc)
        # print(im.shape)
        if self.is_transform:
            im, wc, bm, alb = self.transform(im, wc, bm, alb)
        return im, wc, bm, alb


    def transform(self, img, wc, bm, alb):
        img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        if img.shape[2] == 4:
            img=img[:,:,:3]
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        # img= np.expand_dims(img,-1)

        # img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        alb = m.imresize(alb, (self.bm_size[0], self.bm_size[1])) # uint8 with RGB mode
        alb = alb[:, :, ::-1] # RGB -> BGR
        alb = alb.astype(np.float64)
        if alb.shape[2] == 4:
            alb=alb[:,:,:3]
        # img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            alb = alb.astype(float) / 255.0
        # NHWC -> NCHW
        alb = alb.transpose(2, 0, 1)

        # f, axarr = plt.subplots(2)
        # axarr[0].imshow(wc)
        # axarr[1].imshow(img.transpose(1,2,0))        
        msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)*255
        #normalize label
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67187124
        wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
        wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
        wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)
        wc=cv2.bitwise_and(wc,wc,mask=msk)
        
        wc = m.imresize(wc, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            wc = wc.astype(float) / 255.0
        # NHWC -> NCHW
        wc = wc.transpose(2, 0, 1)

        bm = bm.astype(float)
        #normalize label [-1,1]
        xmx, xmn, ymx, ymn = 435.14545757153445, 13.410177297916455, 435.3297804574046, 14.194541402379988
        bm[:,:,0]= (bm[:,:,0]-xmn)/(xmx-xmn)
        bm[:,:,1]= (bm[:,:,1]-ymn)/(ymx-ymn)
        # bm=bm/np.array([156.0, 187.0])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(self.bm_size[0],self.bm_size[1]))
        bm1=cv2.resize(bm[:,:,1],(self.bm_size[0],self.bm_size[1]))
        

        lbl=np.stack([bm0,bm1],axis=-1)
        # print img.shape

        img = torch.from_numpy(img).float()
        wc = torch.from_numpy(wc).float()
        alb = torch.from_numpy(alb).float()
        lbl = torch.from_numpy(lbl).float()

        return img, wc, lbl, alb


 
# #Leave code for debugging purposes
# #import src.augmentations as aug
# if __name__ == '__main__':
#     local_path = get_data_path('dewarpnet')
#     bs = 4
#     #augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
#     dst = dewarpnete2eLoader(root=local_path, split='valBmMeshsplitnew', is_transform=True)
#     trainloader = data.DataLoader(dst, batch_size=bs)
#     for i, data in enumerate(trainloader):
#         imgs, wcs, bms, albs = data
#         # print imgs.shape
#         imgs = imgs.numpy()
#         imgs = np.transpose(imgs, [0,2,3,1])
#         imgs=imgs[:,:,:,::-1]
#         f, axarr = plt.subplots(bs, 5)
#         # print(labels.shape)
#         bms=bms.numpy()
#         albs=albs.numpy()
#         albs = np.transpose(albs, [0,2,3,1])
#         albs=albs[:,:,:,::-1]
#         wcs=wcs.numpy()
#         wcs = np.transpose(wcs, [0,2,3,1])

#         for j in range(bs):
#             # print(np.min(labels[j]))
#             # print imgs[j].shape
#             axarr[j][0].imshow(imgs[j])
#             axarr[j][1].imshow(wcs[j])
#             axarr[j][2].imshow(bms[j][:,:,0])
#             axarr[j][3].imshow(bms[j][:,:,1])
#             axarr[j][4].imshow(albs[j])
#         plt.show()
#         a = raw_input()
#         if a == 'ex':
#             break
#         else:
#             plt.close()
