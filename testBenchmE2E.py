#test end to end benchmark data
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


from src.models import get_model
from src.loader import get_loader, get_data_path
from src.utils import convert_state_dict


def unwarp(img, bm, pred=True):
    w,h=img.shape[0],img.shape[1]
    if pred:
        bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0,:,:,:]
        #folder 1 xmx, xmn, ymx, ymn = 434.8578833991327, 14.898654260467202, 435.0363953546216, 14.515746051497239
        #folder 2 3 4 '434.9877152088082', '14.546402972133514', '435.0591952709043', '14.489902537540008'
        # denormalize
        dxmx, dxmn, dymx, dymn = 434.9877152088082, 14.546402972133514, 435.0591952709043, 14.489902537540008

        bm=(bm/2.0)+0.5
        bm[:,:,0]=(bm[:,:,0]*(dxmx-dxmn)) +dxmn
        bm[:,:,1]=(bm[:,:,1]*(dymx-dymn)) +dymn
        bm=bm/np.array([448.0, 448.0])
        bm=(bm-0.5)*2
        bm0=cv2.resize(bm[:,:,0],(h,w))
        bm1=cv2.resize(bm[:,:,1],(h,w))
        bm0=cv2.blur(bm0,(3,3))
        bm1=cv2.blur(bm1,(3,3))
        bm=np.stack([bm0,bm1],axis=-1)
        print(bm.shape)
        bm=np.expand_dims(bm,0)
        bm=torch.from_numpy(bm).double()
    else: #bm is nparray (h,w,c)
        bm0=cv2.resize(bm[:,:,0],(h,w),interpolation=cv2.INTER_NEAREST)
        bm1=cv2.resize(bm[:,:,1],(h,w),interpolation=cv2.INTER_NEAREST)
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


def test(args,img_path,fname):
    wc_model_file_name = os.path.split(args.wc_model_path)[1]
    wc_model_name = wc_model_file_name[:wc_model_file_name.find('_')]

    bm_model_file_name = os.path.split(args.bm_model_path)[1]
    bm_model_name = bm_model_file_name[:bm_model_file_name.find('_')]

    wc_data_loader = get_loader(args.dataset+'wcg')
    bm_data_loader = get_loader(args.dataset+'bmni')
    wc_data_path = get_data_path(args.dataset)
    bm_data_path = get_data_path(args.dataset)
    wc_loader = wc_data_loader(wc_data_path, is_transform=True, img_norm=args.img_norm)
    bm_loader = bm_data_loader(bm_data_path, is_transform=True, img_norm=args.img_norm)
    wc_n_classes = wc_loader.n_classes
    bm_n_classes = bm_loader.n_classes
                    
    bm_loader.img_size=(128,128)
    wc_loader.img_size=(256,256)
    

    # Setup image
    print("Read Input Image from : {}".format(img_path))
    imgorg = m.imread(img_path+'.png',mode='RGB')
    # plt.imshow(imgorg)
    # plt.show()
    img = m.imresize(imgorg, (wc_loader.img_size[0], wc_loader.img_size[1]))
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    #img= np.expand_dims(img,-1)
    # img -= wc_loader.mean
    img = img.astype(float) / 255.0
    # plt.imshow(img)
    # plt.show()
    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    print(img.shape)
    

    # Predict WC
    # Setup Model
    wc_model = get_model(wc_model_name, wc_n_classes, in_channels=3)
    wc_state = convert_state_dict(torch.load(args.wc_model_path)['model_state'])
    wc_model.load_state_dict(wc_state)
    wc_model.eval()
    # print(model)

    if torch.cuda.is_available():
        wc_model.cuda(1)
        images = Variable(img.cuda(1), volatile=True)
    else:
        images = Variable(img, volatile=True)

    wc_outputs = wc_model(images)
    softp=nn.Softplus()
    htan = nn.Hardtanh(0,1.0) 
    pred_wc = htan(wc_outputs)

    print(pred_wc.shape)

    # prep input for bm prediction
    pred_wc=pred_wc.detach().cpu().numpy()

    print(np.max(pred_wc))
    print(np.min(pred_wc))
    # NCHW->NWHC
    pred_wc=np.transpose(pred_wc,[0,2,3,1])[0]
    print(pred_wc.shape)

    bminp_wc=cv2.resize(pred_wc,(bm_loader.img_size[0],bm_loader.img_size[1]))
    bminp_img=cv2.resize(imgorg,(bm_loader.img_size[0],bm_loader.img_size[1]))
    print(bminp_wc.shape)
    bminp_img = bminp_img[:, :, ::-1]
    bminp_img = bminp_img.astype(np.float64)
    bminp_img = bminp_img.astype(float) / 255.0
    # NHWC -> NCHW
    bminp_img = bminp_img.transpose(2, 0, 1)
    # NHWC -> NCHW
    bminp_wc = bminp_wc.transpose(2, 0, 1)
    
    #xx_channel=np.ones((bm_loader.img_size[0], bm_loader.img_size[1]))
    #xx_range=np.array(range(bm_loader.img_size[0]))
    #xx_range=np.expand_dims(xx_range,-1)
    #xx_coord=xx_channel*xx_range
    #yy_coord=xx_coord.transpose()

    #xx_coord=xx_coord/(bm_loader.img_size[0]-1)
    #yy_coord=yy_coord/(bm_loader.img_size[0]-1)
    #xx_coord=xx_coord*2 - 1
    #yy_coord=yy_coord*2 - 1
    #xx_coord=np.expand_dims(xx_coord,0)
    #yy_coord=np.expand_dims(yy_coord,0)

    # concat image+wc
    bminp = np.concatenate([bminp_img,bminp_wc],axis=0)
    bminp = np.expand_dims(bminp, 0)

    bm_input = torch.from_numpy(bminp).float()
    # predict bm 
    bm_model = get_model(bm_model_name, bm_n_classes, in_channels=3)
    bm_state = convert_state_dict(torch.load(args.bm_model_path)['model_state'])
    bm_model.load_state_dict(bm_state)
    bm_model.eval()
    # print(model)

    if torch.cuda.is_available():
        bm_model.cuda(1)
        bm_input = Variable(bm_input.cuda(1), volatile=True)
    else:
        bm_input = Variable(bm_input, volatile=True)

    outputs_bm = bm_model(bm_input[:,3:,:,:])


    # use a fixed conv to smooth the output
    # smooth_f=(torch.ones(1,1,15,15)/25.0).cuda(device=1)
    # outputs_bm[:,0,:,:] =F.conv2d(outputs_bm[:,0,:,:].unsqueeze(0),smooth_f,padding=7)
    # outputs_bm[:,1,:,:] =F.conv2d(outputs_bm[:,1,:,:].unsqueeze(0),smooth_f,padding=7)
    # print outputs.shape
    pred_bm = outputs_bm.transpose(1, 2).transpose(2, 3)[0,:,:,:]
    pred_bm= pred_bm.detach().cpu().numpy()


    # call unwarp
    uwpred=unwarp(imgorg, outputs_bm, pred=True)

    f, axarr = plt.subplots(1, 4)
    print (bminp.shape)
    bminp=np.transpose(bminp,[0,2,3,1])
    wcs=bminp[:,:,:,3:][0]
    inp=(bminp[:,:,:,:3][0])[:,:,::-1]

    #axarr[0].imshow(pred_wc)
    #axarr[1].imshow(inp)
    #axarr[2].imshow(pred_bm[:,:,0])
    #axarr[3].imshow(pred_bm[:,:,1])
    
    #plt.show()

    # f1, axarr1 = plt.subplots(1, 2)
    # axarr1[0].imshow(imgorg)
    # axarr1[1].imshow(uwpred)
    # plt.show()

    # Save the output
    cv2.imwrite(args.out_path+fname,uwpred[:,:,::-1]*255)



    

    # wc = wc.astype(float)
    # #normalize label
    # xmx, xmn, ymx, ymn,zmx, zmn=5.7531343, -5.46751, 5.8838153, -5.924346, 2.5959237, -2.88385
    # wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
    # wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
    # wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)
    # wc = m.imresize(wc, (loader.img_size[0], loader.img_size[1])) # uint8 with RGB mode

    # # Resize scales images from 0 to 255, thus we need
    # # to divide by 255.0
    # wc = wc.astype(float) / 255.0
    # # NHWC -> NCHW
    # wc = wc.transpose(2, 0, 1)

    # bm = bm.astype(float)
    # #normalize label [-1,1]
    # # xmx, xmn, ymx, ymn=np.max(bm[:,:,0]), np.min(bm[:,:,0]), np.max(bm[:,:,1]), np.min(bm[:,:,1])
    # # bm[:,:,0]= (bm[:,:,0]-xmn)/(xmx-xmn)
    # # bm[:,:,1]= (bm[:,:,1]-ymn)/(ymx-ymn)
    # bm=bm/np.array([156.0, 187.0])
    # bm=(bm-0.5)*2
    # # bm=(1.0-bm)

    # bm0=cv2.resize(bm[:,:,0],(loader.img_size[0],loader.img_size[1]),cv2.INTER_LANCZOS4)
    # bm1=cv2.resize(bm[:,:,1],(loader.img_size[0],loader.img_size[1]),cv2.INTER_LANCZOS4)

    # lbl=np.stack([bm0,bm1],axis=-1)

    # if model_name in ['pspnet', 'icnet', 'icnetBN']:
    #     img = m.imresize(imgorg, (orig_size[0]//2*2+1, orig_size[1]//2*2+1)) # uint8 with RGB mode, resize width and height which are odd numbers
    # else:
    #     img = m.imresize(imgorg, (loader.img_size[0], loader.img_size[1]))
    # img = img[:, :, ::-1]
    # img = img.astype(np.float64)
    # img -= loader.mean
    # img = img.astype(float) / 255.0
    # # NHWC -> NCHW
    # img = img.transpose(2, 0, 1)
    # img = np.concatenate([img,wc],axis=0)
    # img = np.expand_dims(img, 0)
    # img = torch.from_numpy(img).float()

    # # Setup Model
    # model = get_model(model_name, n_classes, in_channels=6)
    # state = convert_state_dict(torch.load(args.model_path)['model_state'])
    # model.load_state_dict(state)
    # model.eval()

    # if torch.cuda.is_available():
    #     model.cuda(0)
    #     images = Variable(img.cuda(0), volatile=True)
    # else:
    #     images = Variable(img, volatile=True)

    # outputs = model(images)
    # # print outputs.shape
    # pred = outputs.transpose(1, 2).transpose(2, 3)[0,:,:,:]
    

    # print(pred.shape)
    # pred= pred.detach().cpu().numpy()
    
    # # pred = misc.imresize(pred, orig_size, interp='nearest') 
    # #pred = F.upsample(pred, size=(loader.img_size[0], loader.img_size[1],1), mode='nearest')
    # # print pred.shape
    # #get the unwarping
    # uwpred=unwarp(imgorg,outputs,pred=True)
    # uworg=unwarp(imgorg,lbl,pred=False)

    # f, axarr = plt.subplots(2, 4)
    # img=img.numpy()
    # img=np.transpose(img,[0,2,3,1])
    # wcs=img[:,:,:,3:][0]
    # inp=(img[:,:,:,:3][0])[:,:,::-1]

    
    # #for j in range(bs):
    #     #print imgs[j].shape
    # axarr[0][0].imshow(imgorg)
    # # axarr[0][1].imshow(wcs)
    # axarr[0][1].imshow(bm0)
    # axarr[0][2].imshow(bm1)
    # axarr[0][3].imshow(uworg)          #todo: show texture
    # axarr[1][0].imshow(imgorg)
    # # axarr[1][1].imshow(wcs)
    # axarr[1][1].imshow(pred[:,:,0])
    # axarr[1][2].imshow(pred[:,:,1])
    # axarr[1][3].imshow(uwpred)          #todo: show dewarp
    # plt.show()



benchmark_dir='/media/hilab/HiLabData/Sagnik/FoldedDocumentDataset/data/DewarpNet/benchmark/crop/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--wc_model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved wc model')
    parser.add_argument('--bm_model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved bm model')

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
    for fname in os.listdir(benchmark_dir):
        if '.jpg' in fname or '.JPG' in fname or '.png' in fname:
            img_path=args.img_path+fname[:-4]
            test(args,img_path,fname)
