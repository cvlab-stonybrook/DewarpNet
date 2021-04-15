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

from tqdm import tqdm
from torch.utils import data

from .augmentationsk import data_aug, tight_crop

class doc3dwcLoader(data.Dataset):
    """
    Loader for world coordinate regression and RGB images
    """
    def __init__(self, root, split='train', is_transform=False,
                 img_size=512, augmentations=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 3   
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        for split in ['train', 'val']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        #self.setup_annotations()
        if self.augmentations:
            self.txpths=[]
            with open(os.path.join(self.root[:-7],'augtexnames.txt'),'r') as f:
                for line in f:
                    txpth=line.strip()
                    self.txpths.append(txpth)


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                # 1/824_8-cp_Page_0503-7Nw0001
        im_path = pjoin(self.root, 'img',  im_name + '.png')  
        lbl_path=pjoin(self.root, 'wc', im_name + '.exr')
        im = m.imread(im_path,mode='RGB')
        im = np.array(im, dtype=np.uint8)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        lbl = np.array(lbl, dtype=np.float)
        if 'val' in self.split:
            im, lbl=tight_crop(im/255.0,lbl)
        if self.augmentations:          #this is for training, default false for validation\
            tex_id=random.randint(0,len(self.txpths)-1)
            txpth=self.txpths[tex_id] 
            tex=cv2.imread(os.path.join(self.root[:-7],txpth)).astype(np.uint8)
            bg=cv2.resize(tex,self.img_size,interpolation=cv2.INTER_NEAREST)
            im,lbl=data_aug(im,lbl,bg)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl


    def transform(self, img, lbl):
        img = m.imresize(img, self.img_size) # uint8 with RGB mode
        if img.shape[-1] == 4:
            img=img[:,:,:3]   # Discard the alpha channel  
        img = img[:, :, ::-1] # RGB -> BGR
        # plt.imshow(img)
        # plt.show()
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1) # NHWC -> NCHW
        lbl = lbl.astype(float)

        #normalize label
        msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497   # calculate from all the wcs
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
        lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
        lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
        lbl = lbl.transpose(2, 0, 1)   # NHWC -> NCHW
        lbl = np.array(lbl, dtype=np.float)

        # to torch
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()

        return img, lbl



# #Leave code for debugging purposes
# if __name__ == '__main__':
#     local_path = './data/DewarpNet/doc3d/'
#     bs = 4
#     dst = doc3dwcLoader(root=local_path, split='trainswat3dmini', is_transform=True, augmentations=True)
#     trainloader = data.DataLoader(dst, batch_size=bs)
#     for i, data in enumerate(trainloader):
#         imgs, labels = data
#         imgs = imgs.numpy()
#         lbls = labels.numpy()
#         imgs = np.transpose(imgs, [0,2,3,1])
#         lbls = np.transpose(lbls, [0,2,3,1])
#         f, axarr = plt.subplots(bs, 2)
#         for j in range(bs):
#             # print imgs[j].shape
#             axarr[j][0].imshow(imgs[j])
#             axarr[j][1].imshow(lbls[j])
#         plt.show()
#         a = raw_input()
#         if a == 'ex':
#             break
#         else:
#             plt.close()
