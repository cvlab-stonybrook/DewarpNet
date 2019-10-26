# loader for backward mapping 
# loads albedo to dewarp
# uses crop as augmentation
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
import hdf5storage as h5
import random

from tqdm import tqdm
from torch.utils import data

class doc3dbmnoimgcLoader(data.Dataset):
    """
    Data loader for the  semantic segmentation dataset.
    """
    def __init__(self, root, split='train', is_transform=False,
                 img_size=512):
        self.root = os.path.expanduser(root)
        # self.altroot='/home/sagnik/DewarpNet/swat3d/'
        self.altroot='/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/swat3d/'
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        for split in ['train', 'val']:
            path = pjoin(self.altroot, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        #self.setup_annotations()


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                 #1/2Xec_Page_453X56X0001.png
        im_path = pjoin(self.altroot, 'img',  im_name + '.png')  
        img_foldr,fname=im_name.split('/')
        recon_foldr='chess48'
        wc_path = pjoin(self.altroot, 'wc' , im_name + '.exr')
        bm_path = pjoin(self.altroot, 'bm' , im_name + '.mat')
        alb_path = pjoin(self.root,'recon',img_foldr,recon_foldr, fname[:-4]+recon_foldr+'0001.png')

        wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        bm = h5.loadmat(bm_path)['bm']
        alb = m.imread(alb_path,mode='RGB')
        if self.is_transform:
            im, lbl = self.transform(wc,bm,alb)
        return im, lbl


    def tight_crop(self, wc, alb):
        msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)
        size=msk.shape
        [y, x] = (msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        wc = wc[miny : maxy + 1, minx : maxx + 1, :]
        alb = alb[miny : maxy + 1, minx : maxx + 1, :]
        
        s = 20
        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), 'constant')
        alb = np.pad(alb, ((s, s), (s, s), (0, 0)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        wc = wc[cy1 : -cy2, cx1 : -cx2, :]
        alb = alb[cy1 : -cy2, cx1 : -cx2, :]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2

        return wc,alb,t,b,l,r


    def transform(self, wc, bm, alb):
        wc,alb,t,b,l,r=self.tight_crop(wc,alb)               #t,b,l,r = is pixels cropped on top, bottom, left, right
        alb = m.imresize(alb, self.img_size) 
        alb = alb[:, :, ::-1] # RGB -> BGR
        alb = alb.astype(np.float64)
        if alb.shape[2] == 4:
            alb=alb[:,:,:3]
        alb = alb.astype(float) / 255.0
        alb = alb.transpose(2, 0, 1) # NHWC -> NCHW
       
        msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)*255
        #normalize label
        xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497
        wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
        wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
        wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)
        wc=cv2.bitwise_and(wc,wc,mask=msk)
        
        wc = m.imresize(wc, self.img_size) 
        wc = wc.astype(float) / 255.0
        wc = wc.transpose(2, 0, 1) # NHWC -> NCHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([448.0-l-r, 448.0-t-b])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(self.img_size[0],self.img_size[1]))
        bm1=cv2.resize(bm[:,:,1],(self.img_size[0],self.img_size[1]))
        
        img=np.concatenate([alb,wc],axis=0)
        lbl=np.stack([bm0,bm1],axis=-1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl


 
# #Leave code for debugging purposes
# if __name__ == '__main__':
#     local_path = './data/DewarpNet/doc3d/'
#     bs = 4
#     dst = doc3dbmnoimgcLoader(root=local_path, split='trainswat3dmini', is_transform=True)
#     trainloader = data.DataLoader(dst, batch_size=bs)
#     for i, data in enumerate(trainloader):
#         imgs, labels = data
#         imgs = imgs.numpy()
#         imgs = np.transpose(imgs, [0,2,3,1])
#         wcs=imgs[:,:,:,3:]
#         inp=(imgs[:,:,:,:3])[:,:,:,::-1]
#         f, axarr = plt.subplots(bs, 4)
#         labels=labels.numpy()
        
#         for j in range(bs):
#             axarr[j][0].imshow(wcs[j])
#             axarr[j][1].imshow(inp[j])
#             axarr[j][2].imshow(labels[j][:,:,0])
#             axarr[j][3].imshow(labels[j][:,:,1])
#         plt.show()
#         a = raw_input()
#         if a == 'ex':
#             break
#         else:
#             plt.close()
