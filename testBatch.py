# test world coord regression in a batch

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

def hist_eq(imgorg):
    im = np.array(imgorg, dtype=np.uint8)
    img_yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return im

def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]
    # Setup image
    print("Read Input Image from : {}".format(args.img_path))

    imgorg = misc.imread(args.img_path+'.png',mode='RGB')[:,:,:3]
    im = hist_eq(imgorg)

    # read images from the dataset
    data_path='/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/images-corrmesh/'
    filenames=['2/34275Xpr_Page_319X224X0001','2/30548Xcp_Page_0085X273X0001','2/30111Xtc_Page_081X259X0001']
               # '2/31362Xns_Page_639X309X0001','2/33711Xcp_Page_0685X265X0001','2/31864Xns_Page_458X58X0001',
               # '2/32548Xns_Page_002X13X0001','2/34952Xny_Page_060X291X0001','2/32718Xny_Page_261X18X0001']

    batch=[]
    for f in filenames:
        #read file 
        curr_img = misc.imread(data_path+f+'.png')[:,:,:3]
        batch.append(curr_img)
    
    crop_img=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/test/34496Xcp_Page_0213X248X0001crop.png')[:,:,:3]
    nocrop_img=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/test/34496Xcp_Page_0213X248X0001.png')[:,:,:3]
    real_crop=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/test3.png')[:,:,:3]
    real_nocrop=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/test3nocrop.png')[:,:,:3]
    real_nocrop2=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/test.png')[:,:,:3]
    real_nocrop3=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/test4.png')[:,:,:3]
    real_crop3=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/test4.png')[:,:,:3]
    real_crop3=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/test4crop.png')[:,:,:3]
    real_nocrop4=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/test5.png')[:,:,:3]
    real_nobg1=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/53_2copy.png',mode='RGB')
    real_nobg2=misc.imread('/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/51_1copy.png',mode='RGB')
    
    # batch.append(hist_eq(crop_img))
    # batch.append(hist_eq(nocrop_img))
    # batch.append(hist_eq(real_nocrop))
    # batch.append(hist_eq(real_crop))
    # batch.append(hist_eq(real_nocrop2))
    # batch.append(hist_eq(real_nocrop3))
    batch.append(hist_eq(real_crop3))
    batch.append(hist_eq(real_nocrop4))
    # print(batch[0].shape)
    batch.append(hist_eq(real_nobg1))
    batch.append(hist_eq(real_nobg2))

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


    # lbl = lbl[:,:,0]
    # batch_lbl=torch.from_numpy(np.expand_dims(np.expand_dims(lbl,0),0)).cuda()

    data_loader = get_loader(args.dataset+'wc')
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes
    
    loader.img_size=(128,128)
    resized_img = misc.imresize(im, (loader.img_size[0], loader.img_size[1]), interp='bilinear')
    # print(resized_img.shape)

    orig_size = imgorg.shape[:-1]
    img = misc.imresize(im, (loader.img_size[0], loader.img_size[1]))
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    # img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    #apply transformations to batch images
    images=img
    for bimg in batch:
        bimg = misc.imresize(bimg, (loader.img_size[0], loader.img_size[1]))
        bimg = bimg[:, :, ::-1]
        bimg = bimg.astype(np.float64)
        # bimg -= loader.mean
        if args.img_norm:
            bimg = bimg.astype(float) / 255.0
        # NHWC -> NCHW
        bimg = bimg.transpose(2, 0, 1)
        bimg = np.expand_dims(bimg, 0)
        # print (bimg.shape)
        images=np.concatenate([images,bimg],axis=0)

    
    print (images.shape)
    inp_images = torch.from_numpy(images).float()

    # Setup Model
    model = get_model(model_name, n_classes,in_channels=3)

    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    model.train()
    sigm=nn.Sigmoid()
    softp=nn.Softplus(1.0,1.0)
    htan = nn.Hardtanh(0,1.0)

    if torch.cuda.is_available():
        model.cuda(0)
        inp_images = Variable(inp_images.cuda(0), volatile=True)
    else:
        inp_images = Variable(inp_images, volatile=True)

    outputs = model(inp_images)
    print outputs.shape
    target = outputs.transpose(1, 2).transpose(2, 3)
    # target= np.squeeze(target,(0,))
    print target.shape
    #pred=softp(target)
    # batch_pred=softp(outputs)
    batch_pred=htan(outputs)
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

    if args.label:
        f, axarr = plt.subplots(3,batch_size)
    else:
        f, axarr = plt.subplots(2,batch_size)

    for i in range(batch_size): 
        pred=batch_pred[i]
        pred= pred.detach().cpu().numpy()
        pred=pred.transpose(1,2,0)
        print(pred.shape)   
        pred = misc.imresize(pred, loader.img_size, interp='nearest')/255.0
        print(np.max(pred[:,:,0]))
        print(np.max(pred[:,:,1]))
        print(np.max(pred[:,:,2]))
        print(np.min(pred[:,:,0]))
        print(np.min(pred[:,:,1]))
        print(np.min(pred[:,:,2]))
        #pred = F.upsample(pred, size=(loader.img_size[0], loader.img_size[1],1), mode='nearest')
        # print pred.shape
        # print resized_img.shape
        inp=images[i]
        inp=np.transpose(inp,(1,2,0))
        inp=inp[:,:,::-1]        

        #for j in range(bs):
            #print imgs[j].shape
        axarr[0][i].imshow(pred)
        # axarr[1].imshow(disp_surf_norm_pred)
        axarr[1][i].imshow(inp)
        if args.label:
            axarr[2][i].imshow(lbl)
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
