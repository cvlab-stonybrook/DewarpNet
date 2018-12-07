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

class foldeddocwcLoader(data.Dataset):
    """
    Data loader for the  semantic segmentation dataset.
    """
    def __init__(self, root, split='traindewarp', is_transform=False,
                 img_size=512, augmentations=None, img_norm=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 3
        # self.mean = np.array([117.72199, 113.61581, 109.92113])
        self.mean = np.array([0.0, 0.0, 0.0])
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) \
                                               else (img_size, img_size)
        for split in ['traindewarp', 'valdewarp']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        #self.setup_annotations()
        self.tex_root=''   #TODO:


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, 'images',  im_name + '.png')\
        # 3 labels
        lbl1_name = 'WCX'+im_name+'.png'               #1Xec_Page_231X2X0001 WCX1Xec_Page_231X2X0001 DCX1Xec_Page_231X2X0001
        lbl1_path = pjoin(self.root, 'wc',lbl1_name)

        lbl2_name = 'DCX'+im_name+'.png'               
        lbl2_path = pjoin(self.root, 'iuv',lb2_name)

		lbl3_name = im_name.split('X')[1]
        lbl3_path = pjoin(self.tex_root, lb3_name)        

        # print(im_path)
        im = m.imread(im_path)
        im = np.array(im, dtype=np.uint8)
        img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        lbl1 = cv2.imread(lbl1_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        lbl2 = cv2.imread(lbl2_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        lbl3 = cv2.imread(lbl3_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        #lbl = np.array(lbl, dtype=np.float)
        # if self.augmentations is not None:
        #     im, lbl = self.augmentations(im, lbl)
        # print(im.shape)
        if self.is_transform:
            im, lbl1,lbl2,lbl3 = self.transform(im, lbl1,lbl2,lbl3)
        return im, lbl1,lbl2,lbl3


    def transform(self, img, wc,iuv,tex):
        img = m.imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        if img.shape[2] == 4:
			img=img[:,:,:3]
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        #lbl[lbl==255] = 0
        #lbl = lbl.astype(float)
        #print lbl.shape
        wc = wc.astype(float) / 255.0
        wc = cv2.resize(wc, (self.img_size[0], self.img_size[1]), cv2.INTER_NEAREST)
        #lbl = lbl.astype(int)
        iuv = iuv.astype(float) / 255.0
        iuv = cv2.resize(iuv, (self.img_size[0], self.img_size[1]), cv2.INTER_NEAREST)
        tex = tex.astype(float) / 255.0
        tex = cv2.resize(tex, (self.img_size[0], self.img_size[1]), cv2.INTER_NEAREST)

        wc = np.array(wc, dtype=np.float)
        iuv = np.array(iuv, dtype=np.float)
        tex = np.array(tex, dtype=np.float)

        img = torch.from_numpy(img).float()
        wc = torch.from_numpy(wc).float()
        iuv = torch.from_numpy(iuv).float()
        tex = torch.from_numpy(tex).float()
        return img, wc, iuv, tex



# #Leave code for debugging purposes
# #import ptsemseg.augmentations as aug
# if __name__ == '__main__':
#     local_path = get_data_path('foldeddoc')
#     bs = 4
#     #augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
#     dst = foldeddocsphLoader(root=local_path, split='valWc', is_transform=True)
#     trainloader = data.DataLoader(dst, batch_size=bs)
#     for i, data in enumerate(trainloader):
#         imgs, labels = data
#         imgs = imgs.numpy()[:, ::-1, :, :]
#         imgs = np.transpose(imgs, [0,2,3,1])
#         f, axarr = plt.subplots(bs, 2)
#         for j in range(bs):
#             print imgs[j].shape
#             axarr[j][0].imshow(imgs[j])
#             axarr[j][1].imshow(labels[j])
#         plt.show()
#         a = raw_input()
#         if a == 'ex':
#             break
#         else:
#             plt.close()
