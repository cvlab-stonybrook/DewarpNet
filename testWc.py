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
import depthToNormal


from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]
    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    imgorg = misc.imread(args.img_path)
    im_name=os.path.split(args.img_path)[1]
    lbl_name = im_name.split('_')
    lbl_name = lbl_name[0]+'WC_'+lbl_name[1]
    lbl_path = os.path.join('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DepthTrain/', 'worldCoordBVarCam',lbl_name) #DepthMapEXR_10001.exr
    print(lbl_path)
    lbl = np.array(cv2.imread(lbl_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH))

    # lbl = lbl[:,:,0]
    # batch_lbl=torch.from_numpy(np.expand_dims(np.expand_dims(lbl,0),0)).cuda()

    data_loader = get_loader(args.dataset+'wc')
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes
    
    loader.img_size=(284,284)
    resized_img = misc.imresize(imgorg, (loader.img_size[0], loader.img_size[1]), interp='bilinear')

    orig_size = imgorg.shape[:-1]
    if model_name in ['pspnet', 'icnet', 'icnetBN']:
        img = misc.imresize(imgorg, (orig_size[0]//2*2+1, orig_size[1]//2*2+1)) # uint8 with RGB mode, resize width and height which are odd numbers
    else:
        img = misc.imresize(imgorg, (loader.img_size[0], loader.img_size[1]))
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model = get_model(model_name, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()
    sigm=nn.Sigmoid()
    softp=nn.Softplus()

    if torch.cuda.is_available():
        model.cuda(0)
        images = Variable(img.cuda(0), volatile=True)
    else:
        images = Variable(img, volatile=True)

    outputs = model(images)
    print outputs.shape
    target = outputs.transpose(1, 2).transpose(2, 3)[0]
    # target= np.squeeze(target,(0,))
    print target.shape
    pred=softp(target)
    batch_pred=softp(outputs)
    (batch_size, channel, h, w) = batch_pred.size()
    # # get surface normals
    # if pred.is_cuda:
    #     d2n=depthToNormal.DepthToNormals3by3Four(cuda=True)
    # else:
    #     d2n=depthToNormal.DepthToNormals3by3Four(cuda=False)

    # #get grid
    # basegxy = depthToNormal.getBaseGridForCoord(N=(h,w), getbatch = True, batchSize = batch_size)
    # basegxy_orgsize = depthToNormal.getBaseGridForCoord(N=orig_size, getbatch = True, batchSize = batch_size) ### grid for label


    # if pred.is_cuda:
    #     cx = ((basegxy[:,0,:,:].unsqueeze(1)+1)/2).cuda()
    #     cy = ((basegxy[:,1,:,:].unsqueeze(1)+1)/2).cuda()
    #     cx_orgsize = ((basegxy_orgsize[:,0,:,:].unsqueeze(1)+1)/2).cuda()
    #     cy_orgsize = ((basegxy_orgsize[:,1,:,:].unsqueeze(1)+1)/2).cuda()        
    # else:
    #     cx = (basegxy[:,0,:,:].unsqueeze(1)+1)/2
    #     cy = (basegxy[:,1,:,:].unsqueeze(1)+1)/2

    # surface_norms_pred=d2n(cx,cy,batch_pred)
    # surface_norms_lbl=d2n(cx_orgsize,cy_orgsize,batch_lbl)
    # # make unit vectors
    # disp_surf_norm_pred=surface_norms_pred/torch.sqrt(torch.sum(torch.mul(surface_norms_pred,surface_norms_pred),dim=1))
    # disp_surf_norm_pred=disp_surf_norm_pred.squeeze(0).transpose(0,1).transpose(1,2).detach().cpu().numpy()

    # disp_surf_norm_lbl=surface_norms_lbl/torch.sqrt(torch.sum(torch.mul(surface_norms_lbl,surface_norms_lbl),dim=1))
    # disp_surf_norm_lbl=disp_surf_norm_lbl.squeeze(0).transpose(0,1).transpose(1,2).detach().cpu().numpy()

    pred= np.squeeze(pred,0).detach().cpu().numpy()
    pred = misc.imresize(pred, orig_size, interp='nearest') 
    #pred = F.upsample(pred, size=(loader.img_size[0], loader.img_size[1],1), mode='nearest')
    print pred.shape
    print resized_img.shape
    f, axarr = plt.subplots(1, 3)
    #for j in range(bs):
        #print imgs[j].shape
    axarr[0].imshow(pred)
    # axarr[1].imshow(disp_surf_norm_pred)
    axarr[1].imshow(imgorg)
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
    args = parser.parse_args()
    test(args)
