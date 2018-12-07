import sys, os
import torch
import visdom
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
# import depthToNormal


from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

dummy_im_name='2/30008Xpp_Page_225X32X0001'

def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]
    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    imgorg = misc.imread(args.img_path+'.png')[:,:,:3]
    im = np.array(imgorg, dtype=np.uint8)
    img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    im_name=os.path.split(args.img_path)[1]
    if args.label:
        lbl_name = im_name.strip().split('X')
        lbl_name = 'WCX'+lbl_name[0]+lbl_name[3]+'.exr'         #WCX20001.exr
        lbl_path = os.path.join('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/', 'wc-corrmesh/2/',lbl_name) #DepthMapEXR_10001.exr
        print(lbl_path)
        lbl = np.array(cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))
        print (lbl.shape)

        lbl = lbl.astype(float)
        #normalize label
        xmx, xmn, ymx, ymn,zmx, zmn=5.7531343, -5.46751, 5.8838153, -5.924346, 2.5959237, -2.88385
        lbl[:,:,0]= (lbl[:,:,0]-zmn)/(zmx-zmn)
        lbl[:,:,1]= (lbl[:,:,1]-ymn)/(ymx-ymn)
        lbl[:,:,2]= (lbl[:,:,2]-xmn)/(xmx-xmn)


    # read an image and its mask
    dummy_img=cv2.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/images-corrmesh/'+dummy_im_name+'.png').astype(np.uint8)
    dummy_name_split = dummy_im_name.strip().split('X')
    foldr,msk_id=dummy_name_split[0].split('/')
    msk_name = 'MSX'+msk_id+dummy_name_split[3]+'.png'         #WCX20001.exr

    #read the mask
    msk=cv2.imread(os.path.join('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/','mask-corrmesh',foldr,msk_name),0)
    msk=msk.astype(np.uint8)
    
    im=cv2.resize(im,(dummy_img.shape[1],dummy_img.shape[0]),cv2.INTER_LANCZOS4).astype(np.uint8)
    # fg of real
    fg=cv2.bitwise_and(im,im,mask=msk)
    # bg of synth
    rmsk=255-msk
    
    bg=cv2.bitwise_and(dummy_img,dummy_img,mask=rmsk)
    # new_img
    im= cv2.bitwise_xor(bg,fg)
    plt.imshow(im)
    plt.show()

    # lbl = lbl[:,:,0]
    # batch_lbl=torch.from_numpy(np.expand_dims(np.expand_dims(lbl,0),0)).cuda()

    data_loader = get_loader(args.dataset+'wc')
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes
    
    loader.img_size=(284,284)
    resized_img = misc.imresize(im, (loader.img_size[0], loader.img_size[1]), interp='bilinear')
    print(resized_img.shape)

    orig_size = imgorg.shape[:-1]
    if model_name in ['pspnet', 'icnet', 'icnetBN']:
        img = misc.imresize(im, (orig_size[0]//2*2+1, orig_size[1]//2*2+1)) # uint8 with RGB mode, resize width and height which are odd numbers
    else:
        img = misc.imresize(im, (loader.img_size[0], loader.img_size[1]))
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    print (img)
    # Setup Model
    model = get_model(model_name, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    print(model)
    model.eval()
    sigm=nn.Sigmoid()
    softp=nn.Softplus()

    if torch.cuda.is_available():
        model.cuda(0)
        images = Variable(img.cuda(0), volatile=True)
    else:
        images = Variable(img, volatile=True)

    outputs = model(images)
    print outputs
    target = outputs.transpose(1, 2).transpose(2, 3)[0]
    # target= np.squeeze(target,(0,))
    print target.shape
    pred=softp(target)
    batch_pred=softp(outputs)
    (batch_size, channel, h, w) = batch_pred.size()


    pred= np.squeeze(pred,0).detach().cpu().numpy()
    print(np.max(pred[:,:,0]))
    print(np.max(pred[:,:,1]))
    print(np.max(pred[:,:,2]))
    print(np.min(pred[:,:,0]))
    print(np.min(pred[:,:,1]))
    print(np.min(pred[:,:,2]))
    pred = misc.imresize(pred, orig_size, interp='nearest') /255.0
    #pred = F.upsample(pred, size=(loader.img_size[0], loader.img_size[1],1), mode='nearest')
    print pred.shape
    print resized_img.shape
    if args.label:
        f, axarr = plt.subplots(1, 3)
    else:
        f, axarr = plt.subplots(1, 2)

    #for j in range(bs):
        #print imgs[j].shape
    axarr[0].imshow(pred)
    # axarr[1].imshow(disp_surf_norm_pred)
    axarr[1].imshow(resized_img)
    if args.label:
        axarr[2].imshow(lbl)
    # axarr[4].imshow(disp_surf_norm_lbl)
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
    parser.add_argument('--label', dest='label', action='store_true', 
                        help='if your image is from dataset')
    args = parser.parse_args()
    test(args)
