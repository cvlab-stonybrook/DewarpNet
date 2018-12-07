#test backward mapping and unwarp

import sys, os
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as m
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import hdf5storage as h5

# import depthToNormal


from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict


def unwarp(img, bm, pred=True):
    w,h=img.shape[0],img.shape[1]
    if pred:
        bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0,:,:,:]
        # denormalize
        dxmx, dxmn, dymx, dymn = 160.398227753507, -4.444446653356195, 192.11541842533654, -2.922456743468434

        bm=(bm/2.0)+0.5
        bm[:,:,0]=(bm[:,:,0]*(dxmx-dxmn)) +dxmn
        bm[:,:,1]=(bm[:,:,1]*(dymx-dymn)) +dymn
        bm=bm/np.array([156.0, 187.0])
        bm=(bm-0.5)*2
        bm0=cv2.resize(bm[:,:,0],(h,w),cv2.INTER_LANCZOS4)
        bm1=cv2.resize(bm[:,:,1],(h,w),cv2.INTER_LANCZOS4)
        bm=np.stack([bm0,bm1],axis=-1)
        print(bm.shape)
        bm=np.expand_dims(bm,0)
        bm=torch.from_numpy(bm).double()
    else: #bm is nparray (h,w,c)
        bm0=cv2.resize(bm[:,:,0],(h,w),cv2.INTER_LANCZOS4)
        bm1=cv2.resize(bm[:,:,1],(h,w),cv2.INTER_LANCZOS4)
        bm=np.stack([bm0,bm1],axis=-1)
        print(bm.shape)
        bm=np.expand_dims(bm,0)
        bm=torch.from_numpy(bm).double()

    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).double()


    res = F.grid_sample(input=img, grid=bm)
    res = res[0].numpy().transpose((1, 2, 0))
    # print(res.shape)

    # plt.imshow(res)
    # plt.show()
    return res


def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    data_loader = get_loader(args.dataset+'bm')
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes
    
    loader.img_size=(128,128)
    

    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    imgorg = m.imread(args.img_path+'.png')[:,:,:3]
    foldr,im_name=args.img_path.split('/')[-2:]
    lbl_name = im_name.strip().split('X')
    wc_name = 'WCX'+lbl_name[0]+lbl_name[3]+'.exr'
    wc_path= os.path.join('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/','wc-corrmesh',foldr,wc_name)

    bm_name = 'DCX'+lbl_name[0]+lbl_name[3]+'.mat'         #DCX20001.mat
    bm_path = os.path.join('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/', 'bm-corrmesh',foldr,bm_name)

    wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    bm = h5.loadmat(bm_path)['bm']
    
    msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)*255

    wc = wc.astype(float)
    #normalize label
    xmx, xmn, ymx, ymn,zmx, zmn=5.672155, -5.657737, 5.984079, -5.8917637, 1.599145, -3.443543
    wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
    wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
    wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)
    wc=cv2.bitwise_and(wc,wc,mask=msk)
    wc = m.imresize(wc, (loader.img_size[0], loader.img_size[1])) # uint8 with RGB mode

    # Resize scales images from 0 to 255, thus we need
    # to divide by 255.0
    wc = wc.astype(float) / 255.0
    # NHWC -> NCHW
    wc = wc.transpose(2, 0, 1)

    bm = bm.astype(float)
    #normalize label [-1,1]
    # xmx, xmn, ymx, ymn=np.max(bm[:,:,0]), np.min(bm[:,:,0]), np.max(bm[:,:,1]), np.min(bm[:,:,1])
    # bm[:,:,0]= (bm[:,:,0]-xmn)/(xmx-xmn)
    # bm[:,:,1]= (bm[:,:,1]-ymn)/(ymx-ymn)
    bm=bm/np.array([156.0, 187.0])
    bm=(bm-0.5)*2
    # bm=(1.0-bm)

    bm0=cv2.resize(bm[:,:,0],(loader.img_size[0],loader.img_size[1]),cv2.INTER_LANCZOS4)
    bm1=cv2.resize(bm[:,:,1],(loader.img_size[0],loader.img_size[1]),cv2.INTER_LANCZOS4)

    lbl=np.stack([bm0,bm1],axis=-1)

    if model_name in ['pspnet', 'icnet', 'icnetBN']:
        img = m.imresize(imgorg, (orig_size[0]//2*2+1, orig_size[1]//2*2+1)) # uint8 with RGB mode, resize width and height which are odd numbers
    else:
        img = m.imresize(imgorg, (loader.img_size[0], loader.img_size[1]))
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.concatenate([img,wc],axis=0)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model = get_model(model_name, n_classes, in_channels=6)
    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()

    if torch.cuda.is_available():
        model.cuda(0)
        images = Variable(img.cuda(0), volatile=True)
    else:
        images = Variable(img, volatile=True)

    outputs = model(images)
    # print outputs.shape
    pred = outputs.transpose(1, 2).transpose(2, 3)[0,:,:,:]
    

    print(pred.shape)
    pred= pred.detach().cpu().numpy()
    
    # pred = misc.imresize(pred, orig_size, interp='nearest') 
    #pred = F.upsample(pred, size=(loader.img_size[0], loader.img_size[1],1), mode='nearest')
    # print pred.shape
    #get the unwarping
    uwpred=unwarp(imgorg,outputs,pred=True)
    uworg=unwarp(imgorg,lbl,pred=False)

    f, axarr = plt.subplots(2, 4)
    img=img.numpy()
    img=np.transpose(img,[0,2,3,1])
    wcs=img[:,:,:,3:][0]
    inp=(img[:,:,:,:3][0])[:,:,::-1]

    
    #for j in range(bs):
        #print imgs[j].shape
    axarr[0][0].imshow(imgorg)
    # axarr[0][1].imshow(wcs)
    axarr[0][1].imshow(bm0)
    axarr[0][2].imshow(bm1)
    axarr[0][3].imshow(uworg)          #todo: show texture
    axarr[1][0].imshow(imgorg)
    # axarr[1][1].imshow(wcs)
    axarr[1][1].imshow(pred[:,:,0])
    axarr[1][2].imshow(pred[:,:,1])
    axarr[1][3].imshow(uwpred)          #todo: show dewarp
    plt.show()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--img_path', nargs='?', type=str, default=None, 
                        help='Path of the input image')
    # parser.add_argument('--label_path', nargs='?', type=str, default=None, 
    #                   help='Path of the label image')
    parser.add_argument('--out_path', nargs='?', type=str, default=None, 
                        help='Path of the output segmap')
    args = parser.parse_args()
    test(args)
