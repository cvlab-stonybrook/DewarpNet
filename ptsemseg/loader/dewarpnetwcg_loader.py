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

from augmentations_gray import call_augmentations

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

class dewarpnetwcgLoader(data.Dataset):
    """
    Data loader for the  semantic segmentation dataset.
    """
    def __init__(self, root, split='trainBmMeshsplitnew', is_transform=False,
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
        for split in ['trainBmMeshsplitnew', 'valBmMeshsplitnew']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        #self.setup_annotations()


    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        im_name = self.files[self.split][index]                 #1/2Xec_Page_453X56X0001.png
        im_path = pjoin(self.root, 'images-corrmesh',  im_name + '.png')  
        lbl_name = im_name.strip().split('X')
        foldr,lbl_id=lbl_name[0].split('/')
        lbl_name = 'WCX'+lbl_id+lbl_name[3]+'.exr'         #WCX20001.exr
        lbl_path = pjoin(self.root, 'wc-corrmesh',foldr,lbl_name) 

        # print(im_path)
        im = m.imread(im_path,mode='RGB')
        im = np.array(im, dtype=np.uint8)
        # img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)

        # # equalize the histogram of the Y channel
        # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        # im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        lbl = cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        #lbl = np.array(lbl, dtype=np.float)
        if self.augmentations:
            im, lbl = call_augmentations(self.root,im_name,im, lbl)
        # print(im.shape)
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
        #print lbl.shape
        lbl = lbl.astype(float)
        #normalize label
        msk=((lbl[:,:,0]!=0)&(lbl[:,:,1]!=0)&(lbl[:,:,2]!=0)).astype(np.uint8)*255
        # print(np.max(msk))
        # print(np.min(msk))

        # plt.imshow(msk)
        # plt.show()

        ####################################### Uncomment when not just mask
        xmx, xmn, ymx, ymn,zmx, zmn=5.672155, -5.657737, 5.984079, -5.8917637, 1.599145, -3.443543
        # xmx, xmn, ymx, ymn,zmx, zmn=6.267521, -6.2670546, 5.0372734, -5.039255, 2.646455, -2.6430461
        
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)
        lbl=cv2.bitwise_and(lbl,lbl,mask=msk)
        ######################################

    
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), cv2.INTER_NEAREST)
        #lbl = lbl.astype(int)
        # print (lbl.shape)
        # msk=expand_dims(msk/255.0,-1)

        lbl = np.array(lbl, dtype=np.float)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        # msk = torch.from_numpy(msk).float()
        # print (img.shape)

        return img, lbl


    # def get_pascal_labels(self):
    #     """Load the mapping that associates pascal classes with label colors

    #     Returns:
    #         np.ndarray with dimensions (21, 3)
    #     """
    #     return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0],
    #                       [0,0,128], [128,0,128], [0,128,128], [128,128,128],
    #                       [64,0,0], [192,0,0], [64,128,0], [192,128,0],
    #                       [64,0,128], [192,0,128], [64,128,128], [192,128,128],
    #                       [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
    #                       [0,64,128]])


    # def encode_segmap(self, mask):
    #     """Encode segmentation label images as pascal classes

    #     Args:
    #         mask (np.ndarray): raw segmentation label image of dimension
    #           (M, N, 3), in which the Pascal classes are encoded as colours.

    #     Returns:
    #         (np.ndarray): class map with dimensions (M,N), where the value at
    #         a given location is the integer denoting the class index.
    #     """
    #     mask = mask.astype(int)
    #     label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    #     for ii, label in enumerate(self.get_pascal_labels()):
    #         label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    #     label_mask = label_mask.astype(int)
    #     return label_mask


    # def decode_segmap(self, label_mask, plot=False):
    #     """Decode segmentation class labels into a color image

    #     Args:
    #         label_mask (np.ndarray): an (M,N) array of integer values denoting
    #           the class label at each spatial location.
    #         plot (bool, optional): whether to show the resulting color image
    #           in a figure.

    #     Returns:
    #         (np.ndarray, optional): the resulting decoded color image.
    #     """
    #     label_colours = self.get_pascal_labels()
    #     r = label_mask.copy()
    #     g = label_mask.copy()
    #     b = label_mask.copy()
    #     for ll in range(0, self.n_classes):
    #         r[label_mask == ll] = label_colours[ll, 0]
    #         g[label_mask == ll] = label_colours[ll, 1]
    #         b[label_mask == ll] = label_colours[ll, 2]
    #     rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    #     rgb[:, :, 0] = r / 255.0
    #     rgb[:, :, 1] = g / 255.0
    #     rgb[:, :, 2] = b / 255.0
    #     if plot:
    #         plt.imshow(rgb)
    #         plt.show()
    #     else:
    #         return rgb

    # def setup_annotations(self):
    #     """Sets up Berkley annotations by adding image indices to the
    #     `train_aug` split and pre-encode all segmentation labels into the
    #     common label_mask format (if this has not already been done). This
    #     function also defines the `train_aug` and `train_aug_val` data splits
    #     according to the description in the class docstring
    #     """
    #     sbd_path = get_data_path('foldeddoc')
    #     target_path = pjoin(self.root, 'SegmentationClass/pre_encoded')
    #     if not os.path.exists(target_path): os.makedirs(target_path)
    #     path = pjoin(sbd_path, 'train.txt')
    #     sbd_train_list = tuple(open(path, 'r'))
    #     sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
    #     train_aug = self.files['train'] + sbd_train_list

    #     # keep unique elements (stable)
    #     train_aug = [train_aug[i] for i in \
    #                       sorted(np.unique(train_aug, return_index=True)[1])]
    #     self.files['train_aug'] = train_aug
    #     set_diff = set(self.files['val']) - set(train_aug) # remove overlap
    #     self.files['train_aug_val'] = list(set_diff)

    #     pre_encoded = glob.glob(pjoin(target_path, '*.png'))
    #     expected = np.unique(self.files['train_aug'] + self.files['val']).size

    #     if len(pre_encoded) != expected:
    #         print("Pre-encoding segmentation masks...")
    #         for ii in tqdm(sbd_train_list):
    #             lbl_path = pjoin(sbd_path, 'dataset/cls', ii + '.mat')
    #             data = io.loadmat(lbl_path)
    #             lbl = data['GTcls'][0]['Segmentation'][0].astype(np.int32)
    #             lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
    #             m.imsave(pjoin(target_path, ii + '.png'), lbl)

    #         for ii in tqdm(self.files['trainval']):
    #             fname = ii + '.png'
    #             lbl_path = pjoin(self.root, 'SegmentationClass', fname)
    #             lbl = self.encode_segmap(m.imread(lbl_path))
    #             lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
    #             m.imsave(pjoin(target_path, fname), lbl)

    #     assert expected == 9733, 'unexpected dataset sizes'

# #Leave code for debugging purposes
# #import ptsemseg.augmentations as aug
# if __name__ == '__main__':
#     local_path = get_data_path('dewarpnet')
#     bs = 4
#     #augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip()])
#     dst = dewarpnetwcLoader(root=local_path, split='trainnewmesh', is_transform=True, augmentations=False)
#     trainloader = data.DataLoader(dst, batch_size=bs)
#     for i, data in enumerate(trainloader):
#         imgs, labels = data
#         imgs = imgs.numpy()
#         imgs = np.transpose(imgs, [0,2,3,1])
#         f, axarr = plt.subplots(bs, 2)
#         for j in range(bs):
#             print imgs[j].shape
#             axarr[j][0].imshow(imgs[j][:,:,0],cmap='gray')
#             axarr[j][1].imshow(labels[j],cmap='gray')
#             # print(np.min(labels[j].numpy()[:,:,0]))
#             # print(np.min(labels[j].numpy()[:,:,1]))
#             # print(np.min(labels[j].numpy()[:,:,2]))
#         plt.show()
#         a = raw_input()
#         if a == 'ex':
#             break
#         else:
#             plt.close()
