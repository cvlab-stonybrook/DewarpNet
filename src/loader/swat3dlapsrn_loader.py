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

class swat3dlapsrnLoader(data.Dataset):
    """
    Data loader for the lapsrn on world coordinate outputs
    """
    def __init__(self, root, split='trainswat3d', is_transform=False,
                 img_size=512, lbl_size=224, augmentations=None, img_norm=True):
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
        self.lbl_size = lbl_size if isinstance(lbl_size, tuple) \
                                               else (lbl_size, lbl_size)

        # self.model = src.models.get_model(model_name,n_classes=self.n_classes, in_channels=3)
        # self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        # self.model.cuda()

        # if model_path:
        #     checkpoint = torch.load(model_path)
        #     self.model.load_state_dict(checkpoint['model_state'])

        #     # freeze module1 parameters
        #     for param in self.model.parameters():
        #         param.requires_grad=False
        # else:
        #     print("Please enter a checkpoint...")


        for split in ['trainswat3d', 'valswat3d']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
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
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        if self.is_transform:
            im, lbl1,lbl2 = self.transform(im, lbl)

        # im is the input image in (256,256)
        # return 2 labels of (224,224) and (448,448)
        
        return im, lbl1,lbl2


    def transform(self, img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        if img.shape[2] == 4:
            img=img[:,:,:3]
        img = img[:, :, ::-1] # RGB -> BGR
        # plt.imshow(img)
        # plt.show()
        img = img.astype(np.float64)
        # img= np.expand_dims(img,-1)

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
        
        # swat3d fixed (folder1): 1.2315044, -1.2415468, 1.2283025, -1.2258043, 0.6339816, -0.63410246
        # swat3d varY (folder2): 1.2394472, -1.2386274, 1.228057, -1.2263123, 0.63385504, -0.64252603
        # swat3d varX (folder3): 1.2388626, -1.2405566, 1.2291702, -1.2275329, 0.63167834, -0.6692216
        # swat3d varXY (folder4): 1.2375582, -1.2396094, 1.2277281, -1.2126498, 0.632438, -0.67008704
        # swat3d rand (folder5,6,7): 1.2385225, -1.2410645, 1.2297894, -1.2273213, 0.63452387, -0.67187124

        xmx, xmn, ymx, ymn,zmx, zmn= 1.2394472, -1.2415468, 1.2297894, -1.2275329, 0.63452387, -0.67187124
        
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
        lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
        # plt.imshow(lbl)
        # plt.show()
        # print lbl.shape
        ######################################

    
        lbl1 = cv2.resize(lbl, (self.lbl_size[0], self.lbl_size[1]), interpolation=cv2.INTER_NEAREST) #this is (224,224)
        lbl2 = lbl #this is hres (448,448) 
        #lbl = lbl.astype(int)
        # print (lbl.shape)
        # msk=expand_dims(msk/255.0,-1)

        lbl1 = np.array(lbl1, dtype=np.float)
        lbl2 = np.array(lbl2, dtype=np.float)
        img = torch.from_numpy(img).float()
        lbl1 = torch.from_numpy(lbl1).float()
        lbl2 = torch.from_numpy(lbl2).float()
        # msk = torch.from_numpy(msk).float()
        # print (img.shape)

        return img,lbl1,lbl2



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
