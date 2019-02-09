#loader for world coordinate regression and grayscale images

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

class swat3dwcgLoader(data.Dataset):
    """
    Data loader for the  semantic segmentation dataset.
    """
    def __init__(self, root, split='train', is_transform=False,
                 img_size=512, augmentations=None, img_norm=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 3                                      ##################<----------------- Change classes back to 3 when not mask
        # self.mean = np.array([117.72199, 113.61581, 109.92113])
        self.mean = np.array([0.0, 0.0, 0.0])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
                                               else (img_size, img_size)
        self.train_files=['trainswat3dfixed','trainswat3dvarY']
        self.val_files=['valswat3dfixed','valswat3dvarY']
        for split in ['train', 'val']:
            if split=='train':
                file_list=[]
                for fn in self.train_files:
                    path = pjoin(self.root, fn + '.txt')
                    fl = tuple(open(path, 'r'))
                    fl = [id_.rstrip() for id_ in fl]
                    file_list+=fl
                self.files[split] = file_list
            elif split=='val':
                file_list=[]
                for fn in self.val_files:
                    path = pjoin(self.root, fn + '.txt')
                    fl = tuple(open(path, 'r'))
                    fl = [id_.rstrip() for id_ in fl]
                    file_list+=fl
                self.files[split] = file_list

        #self.setup_annotations()


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                 #1/2Xec_Page_453X56X0001     1/824_8-cp_Page_0503-7Nw0001
        im_path = pjoin(self.root, 'img',  im_name + '.png')  

        ################################################################### This is for Pre CVPR'18 data
        # lbl_name = im_name.strip().split('X')
        # foldr,lbl_id=lbl_name[0].split('/')
        # lbl_name = 'WCX'+lbl_id+lbl_name[3]+'.exr'         #WCX20001.exr
        # lbl_path = pjoin(self.root, 'wc-corrmesh',foldr,lbl_name) 
        ###################################################################
        
        lbl_path=pjoin(self.root, 'wc', im_name + '.exr')
        # print(lbl_path)

        # print(im_path)
        im = m.imread(im_path,mode='RGB')
        im = np.array(im, dtype=np.uint8)
        # img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)

        # # # equalize the histogram of the Y channel
        # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # # convert the YUV image back to RGB format
        # im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        lbl = np.array(lbl, dtype=np.float)
        if self.augmentations:
            im, lbl = call_augmentations(self.root,im_name,im, lbl)
        # print(im.shape)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


    def transform(self, img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        # if img.shape[2] == 4:
        #     img=img[:,:,:3]
        # img = img[:, :, ::-1] # RGB -> BGR
        # plt.imshow(img)
        # plt.show()
        img = img.astype(np.float64)
        img= np.expand_dims(img,-1)

        # img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        #lbl[lbl==255] = 0
        #lbl = lbl.astype(float)
        # print lbl.shape
        lbl = lbl.astype(float)
        #normalize label
        msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
        # print(np.max(msk))
        # print(np.min(msk))

        # plt.imshow(msk)
        # plt.show()

        ####################################### Uncomment when not just mask
        # xmx, xmn, ymx, ymn,zmx, zmn=5.672155, -5.657737, 5.984079, -5.8917637, 1.599145, -3.443543
        # xmx, xmn, ymx, ymn,zmx, zmn=6.267521, -6.2670546, 5.0372734, -5.039255, 2.646455, -2.6430461

        xmx, xmn, ymx, ymn,zmx, zmn= 1.2315044, -1.2415468, 1.2283025, -1.2258043, 0.6339816, -0.63410246
        
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
        lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
        # plt.imshow(lbl)
        # plt.show()
        # print lbl.shape
        ######################################

    
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        #lbl = lbl.astype(int)
        # print (lbl.shape)
        # msk=expand_dims(msk/255.0,-1)

        lbl = np.array(lbl, dtype=np.float)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        # msk = torch.from_numpy(msk).float()
        # print (img.shape)

        return img, lbl



# #Leave code for debugging purposes
# #import src.augmentations as aug
# if __name__ == '__main__':
#     local_path = get_data_path('swat3d')
#     bs = 4
#     #augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
#     dst = swat3dwcgLoader(root=local_path, split='valswat3dfixed', is_transform=True, augmentations=False)
#     trainloader = data.DataLoader(dst, batch_size=bs)
#     for i, data in enumerate(trainloader):
#         imgs, labels = data
#         imgs = imgs.numpy()
#         imgs = np.transpose(imgs, [0,2,3,1])
#         f, axarr = plt.subplots(bs, 2)
#         for j in range(bs):
#             print imgs[j].shape
#             axarr[j][0].imshow(imgs[j][:,:,0],cmap='gray')
#             axarr[j][1].imshow(labels[j])
#             # print(np.min(labels[j].numpy()[:,:,0]))
#             # print(np.min(labels[j].numpy()[:,:,1]))
#             # print(np.min(labels[j].numpy()[:,:,2]))
#         plt.show()
#         a = raw_input()
#         if a == 'ex':
#             break
#         else:
#             plt.close()
