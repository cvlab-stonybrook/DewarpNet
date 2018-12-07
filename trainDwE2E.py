# train dewarpnet in an end to end fashion
# activations on wc,bm regression is hardtanh
# initial input is grayscale image
# bm regression models are stored in checkpoints-e2e: BEST:dnet_dewarpnet_8_0.718680620193_0.859021067619_htan_e2eretrain_ssimreconssiml1_best_model.pkl
# wc regression models are in checkpoints: BEST l1: unetnc_dewarpnet_22_0.00191422272474_0.000375664709754_htan_e2eretrain_l1_best_model.pkl
#                                          BEST other losses: unetnc_dewarpnet_2_0.0019854803104_0.000491506556931_htan_e2eretrain_ssimreconssiml1_best_model.pkl
import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils import data
from torchvision import utils
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
import recon_loss 
import pytorch_ssim



def show_uloss(uwpred,uworg,inp_img):
    n,c,h,w=inp_img.shape
    
    # print(labels.shape)
    uwpred=uwpred.detach().cpu().numpy()
    uworg=uworg.detach().cpu().numpy()
    inp_img=inp_img.detach().cpu().numpy()

    #NCHW->NHWC
    uwpred=uwpred.transpose((0, 2, 3, 1))
    uworg=uworg.transpose((0, 2, 3, 1))

    # f, axarr = plt.subplots(n, 3)
    # for j in range(n):
    #     # print(np.min(labels[j]))
    #     # print imgs[j].shape
    #     img=inp_img[j].transpose(1,2,0)
    #     axarr[j][0].imshow(img[:,:,::-1])
    #     axarr[j][1].imshow(uworg[j])
    #     axarr[j][2].imshow(uwpred[j])
    
    # plt.savefig('./generated/unwarp.png')
    # plt.close()
    # a=input()


def show_uloss_visdom(vis,uwpred,uworg,labels_win,out_win,labelopts,outopts,args):
    n,c,h,w=uwpred.shape
    
    # print(labels.shape)
    uwpred=uwpred.detach().cpu().numpy()
    uworg=uworg.detach().cpu().numpy()
    out_arr=np.full((4,3,args.img_rows,args.img_cols),0.0)
    label_arr=np.full((4,3,args.img_rows,args.img_cols),0.0)
    choices=random.sample(range(n), 4)
    idx=0
    for c in choices:
        out_arr[idx,:,:,:]=uwpred[c]
        label_arr[idx,:,:,:]=uworg[c]
        idx+=1

    vis.images(out_arr,
               win=out_win,
               opts=outopts)
    vis.images(label_arr,
               win=labels_win,
               opts=labelopts)




def train(args):

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset+'e2e')
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='valBmMeshsplitnew', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)


    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        train_labels1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train label 1x', caption='In progress..'))
        train_labels1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train label 1y', caption='In progress..'))
        train_out1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train Output 1x', caption='In progress..'))
        train_out1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train Output 1y', caption='In progress..'))

        val_labels1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val labels 1x', caption='In progress..'))
        val_labels1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val labels 1y', caption='In progress..'))
        val_out1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Outputs 1x', caption='In progress..'))
        val_out1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Outputs 1y', caption='In progress..'))
        
        train_labels_win = vis.images(np.full((4,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train labels', caption='Train GT Dewarp'))
        train_out_win = vis.images(np.full((4,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train Outputs', caption='Train Pred Dewarp'))
        val_labels_win = vis.images(np.full((4,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Labels', caption='Val GT Dewarp'))
        val_out_win = vis.images(np.full((4,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Outputs', caption='Val Pred Dewarp'))
        

    # Setup Model
    wc_model = get_model(args.wc_arch, n_classes=3, in_channels=1)
    bm_model = get_model(args.bm_arch, n_classes=2, in_channels=3)
    
    wc_model = torch.nn.DataParallel(wc_model, device_ids=range(torch.cuda.device_count()))
    wc_model.cuda()

    bm_model = torch.nn.DataParallel(bm_model, device_ids=range(torch.cuda.device_count()))
    bm_model.cuda()
    
    # Setup Optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    optimizer= torch.optim.Adam([{'params':wc_model.parameters()},
                                 {'params':bm_model.parameters()}  ],lr=args.l_rate, weight_decay=5e-4, amsgrad=True)

    # Define Loss
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()
    reconst_loss= recon_loss.Unwarploss()
    ssim_loss=pytorch_ssim.SSIM(window_size=10,channels=2)
    epoch_start=0

    # Look for WC checkpoint 
    if args.wc_resume is not None:                                         
        if os.path.isfile(args.wc_resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.wc_resume))
            checkpoint = torch.load(args.wc_resume)
            wc_model.load_state_dict(checkpoint['model_state'])
            # optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.wc_resume, checkpoint['epoch']))
            # epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.wc_resume)) 

    # Look for BM checkpoint 
    if args.bm_resume is not None:                                         
        if os.path.isfile(args.bm_resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.bm_resume))
            checkpoint = torch.load(args.bm_resume)
            bm_model.load_state_dict(checkpoint['model_state'])
            # optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.bm_resume, checkpoint['epoch']))
            # epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.bm_resume)) 

    # Activations
    htan = nn.Hardtanh(0,1.0)

    best_val_mse =999999.0
    best_val_ssim =0.0

    log_count=0
    for epoch in range(epoch_start,args.n_epoch):
        avg_loss=0.0
        avgl1loss=0.0
        avg_nl1=0.0
        train_mse=0.0
        train_ssim=0.0
        wc_model.train()
        bm_model.train()

        for i, (imgs, wcs, bms, albs) in enumerate(trainloader):
            images = Variable(imgs.cuda())
            wc_labels = Variable(wcs.cuda())
            bm_labels = Variable(bms.cuda())    #this is nhwc
            alb_labels=Variable(albs.cuda())


            optimizer.zero_grad()
            # print (images.shape)
            wc_outputs = wc_model(images)       #this is (nchw)  
            wc_outputs = F.upsample(wc_outputs, size=(args.img_rows, args.img_cols), mode='bilinear')
            # wc_outputs_nhwc = wc_outputs.transpose(1, 2).transpose(2, 3)
            #print(target_nhwc.shape)
            wc_pred_nchw=htan(wc_outputs)
            wc_loss = loss_fn(wc_pred_nchw, wc_labels)

            bm_outputs = bm_model(wc_pred_nchw)
            bm_outputs_nhwc = bm_outputs.transpose(1, 2).transpose(2, 3)
            bm_labels_nchw=bm_labels.transpose(3,2).transpose(2,1)
            l1loss = loss_fn(bm_outputs_nhwc, bm_labels)
            # batch_ssim=ssim_loss(bm_outputs,bm_labels_nchw)
            # ssloss=1-batch_ssim
            # print(alb_labels.shape)
            # print(bm_outputs_nhwc.shape)
            rloss,ssim,uworg,uwpred = reconst_loss(alb_labels,bm_outputs_nhwc,bm_labels)
            alb_labels_nchw=alb_labels.transpose(3, 2).transpose(2, 1)
            bm_loss=(1.0*l1loss)+(0.0*rloss) + (0.0*ssim)
            # show_uloss(uwpred,uworg,images[:,:3,:,:])

            loss=wc_loss+bm_loss
            
            avg_loss+=loss
            
            train_mse+=MSE(wc_pred_nchw, wc_labels).item()
            train_ssim+=(1-ssim)

            loss.backward()
            # for param in model.parameters():
            #     print(param.grad.data.sum())
            optimizer.step()

            if (i+1) % 20 == 0:
                avg_loss=avg_loss/20
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss))
                # print("L1:%4f, SNL1:%.4f" %(avgl1loss.item()/50,avg_nl1.item()/50))
            
            if args.visdom:
                choices=random.sample(range(images.shape[0]), 1)
                #show batch output and labels
                outx_arr=np.full((args.img_rows,args.img_cols),0)
                outy_arr=np.full((args.img_rows,args.img_cols),0)
                labelx_arr=np.full((args.img_rows,args.img_cols),0) 
                labely_arr=np.full((args.img_rows,args.img_cols),0) 
                idx=0
                target_cpu=bm_outputs.detach().cpu().numpy()
                labels_cpu=bm_labels.detach().cpu().numpy()
                for c in choices:
                    # labels_nchw=labels.transpose(3,2).transpose(2,1)
                    # print(labels_nchw.shape)
                    outx_arr=target_cpu[c,0,:,:]
                    outy_arr=target_cpu[c,1,:,:]
                    labelx_arr=labels_cpu[c,:,:,0]
                    labely_arr=labels_cpu[c,:,:,1]
                    # print(np.max(labelx_arr))
                    # print(np.min(labelx_arr))
                    idx+=1
                vis.heatmap(outx_arr,
                           win=train_out1x_win,
                           opts=dict(title='Train Output 1x', caption='In progress..'))
                vis.heatmap(outy_arr,
                           win=train_out1y_win,
                           opts=dict(title='Train Output 1y', caption='In progress..'))
                vis.heatmap(labelx_arr,
                           win=train_labels1x_win,
                           opts=dict(title='Train Label 1x', caption='In progress..'))
                vis.heatmap(labely_arr,
                           win=train_labels1y_win,
                           opts=dict(title='Train Label 1y', caption='In progress..'))
                labelopts=dict(title='Train Label', caption='Gt unwarp')
                outopts=dict(title='Train Out', caption='Pred. unwarp')
                show_uloss_visdom(vis,uwpred,uworg,train_labels_win,train_out_win,labelopts,outopts,args)


        # print("L1:%4f, SNL1:%.4f" %(avgl1loss.item()/len(trainloader),avg_nl1.item()/len(trainloader)))

        train_mse=train_mse/len(trainloader)
        train_ssim=train_ssim/len(trainloader)
        print("Training mse:'{}'".format(train_mse))
        print("Training ssim:'{}'".format(train_ssim))

        wc_model.eval()
        bm_model.eval()
        val_loss=0.0
        wc_val_mse=0.0
        bm_val_ssim=0.0
        for i_val, (imgs_val, wcs_val, bms_val, albs_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(imgs_val.cuda())
                wc_labels_val = Variable(wcs_val.cuda())
                bm_labels_val = Variable(bms_val.cuda())    #this is nhwc
                alb_labels_val =Variable(albs_val.cuda())

                wc_outputs_val = wc_model(images_val)       #this is (nchw)  
                wc_outputs_val = F.upsample(wc_outputs_val, size=(args.img_rows, args.img_cols), mode='bilinear')
                # wc_outputs_nhwc = wc_outputs.transpose(1, 2).transpose(2, 3)
                #print(target_nhwc.shape)
                wc_pred_nchw_val=htan(wc_outputs_val)
                wc_loss_val = loss_fn(wc_pred_nchw_val, wc_labels_val)

                bm_outputs_val = bm_model(wc_pred_nchw_val)
                bm_outputs_nhwc_val = bm_outputs_val.transpose(1, 2).transpose(2, 3)
                bm_labels_nchw_val=bm_labels_val.transpose(3,2).transpose(2,1)
                
                l1loss = loss_fn(bm_outputs_nhwc_val, bm_labels_val)
                # batch_ssim=ssim_loss(bm_outputs_val,bm_labels_nchw_val)
                # ssloss=1-batch_ssim  
                rloss,ssim,uworg,uwpred = reconst_loss(alb_labels_val,bm_outputs_nhwc_val,bm_labels_val)
                alb_labels_nchw_val=alb_labels_val.transpose(3, 2).transpose(2, 1)
                bm_loss_val=(1.0*l1loss)+(0.0*rloss) + (0.0*ssim)
                # show_uloss(uwpred,uworg,images[:,:3,:,:])
                loss=wc_loss_val+bm_loss_val
                wc_pred=wc_pred_nchw_val.data.cpu()
                wc_gt = wc_labels_val.cpu()
                val_loss+=loss
                wc_val_mse+=MSE(wc_pred, wc_gt)
                bm_val_ssim+=(1-ssim)

            if args.visdom:
                choices=random.sample(range(images.shape[0]), 1)
                #show batch output and labels
                outx_arr=np.full((args.img_rows,args.img_cols),0)
                outy_arr=np.full((args.img_rows,args.img_cols),0)
                labelx_arr=np.full((args.img_rows,args.img_cols),0) 
                labely_arr=np.full((args.img_rows,args.img_cols),0) 
                idx=0
                for c in choices:
                    # labels_nchw=labels_val.transpose(3,2).transpose(2,1)
                    # print(labels_nchw.shape)
                    target_cpu=bm_outputs_val.detach().cpu().numpy()
                    labels_cpu=bm_labels_val.detach().cpu().numpy()
                    outx_arr=target_cpu[c,0,:,:]
                    outy_arr=target_cpu[c,1,:,:]
                    labelx_arr=labels_cpu[c,:,:,0]
                    labely_arr=labels_cpu[c,:,:,1]
                    idx+=1
                vis.heatmap(outx_arr,
                           win=val_out1x_win,
                           opts=dict(title='Val Output 1x', caption='In progress..'))
                vis.heatmap(outy_arr,
                           win=val_out1y_win,
                           opts=dict(title='Val Output 1y', caption='In progress..'))
                vis.heatmap(labelx_arr,
                           win=val_labels1x_win,
                           opts=dict(title='Val Label 1x', caption='In progress..'))
                vis.heatmap(labely_arr,
                           win=val_labels1y_win,
                           opts=dict(title='Val Label 1y', caption='In progress..'))
                
                labelopts=dict(title='Val Label', caption='Gt unwarp')
                outopts=dict(title='Val Out', caption='Pred. unwarp')
                show_uloss_visdom(vis,uwpred,uworg,val_labels_win,val_out_win,labelopts,outopts,args)



        print("val loss at epoch {}:: {}".format(epoch+1,val_loss/len(valloader)))
        print("val ssim at epoch {}:: {}".format(epoch+1,bm_val_ssim/len(valloader)))
        wc_val_mse=wc_val_mse/len(valloader)
        bm_val_ssim=bm_val_ssim/len(valloader)
        print("val mse: {}".format(wc_val_mse))        
        if wc_val_mse < best_val_mse:
            best_val_mse=wc_val_mse
            state = {'epoch': epoch+1,
                     'model_state': wc_model.state_dict(),}
            torch.save(state, "./checkpoints/{}_{}_{}_{}_{}_htan_e2eretrain_l1_best_model.pkl".format(args.wc_arch, args.dataset, epoch+1,wc_val_mse,train_mse))
        if bm_val_ssim > best_val_ssim:
            best_val_ssim=bm_val_ssim
            state = {'epoch': epoch+1,
                     'model_state': bm_model.state_dict(),}
            torch.save(state, "./checkpoints-e2e/{}_{}_{}_{}_{}_htan_e2eretrain_l1_best_model.pkl".format(args.bm_arch, args.dataset, epoch+1,bm_val_ssim,train_ssim))

        if (epoch+1) % 10 == 0:
            # best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': wc_model.state_dict(),}
            torch.save(state, "./checkpoints/{}_{}_{}_{}_{}_htan_e2eretrain_l1_model.pkl".format(args.wc_arch, args.dataset, epoch+1,wc_val_mse,train_mse))

            state = {'epoch': epoch+1,
                     'model_state': bm_model.state_dict(),}
            torch.save(state, "./checkpoints-e2e/{}_{}_{}_{}_{}_htan_e2eretrain_l1_model.pkl".format(args.bm_arch, args.dataset, epoch+1,bm_val_ssim,train_ssim))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--wc_arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use for wc regression')
    parser.add_argument('--bm_arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use for backward mapping')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--wc_resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved WC model to restart from')
    parser.add_argument('--bm_resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved BM model to restart from')
    parser.add_argument('--visdom', dest='visdom', action='store_true', 
                        help='Enable visualization(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false', 
                        help='Disable visualization(s) on visdom | False by default')
    parser.set_defaults(visdom=False)

    args = parser.parse_args()
    train(args)


 #CUDA_VISIBLE_DEVICES=1 python trainDwE2E.py --wc_arch unetnc --bm_arch dnet --dataset dewarpnet --img_rows 128 --img_cols 128 --img_norm --n_epoch 100 --batch_size 50 --l_rate 0.0001 --wc_resume ./checkpoints/unetnc_dewarpnet_290_0.00384727632627_0.00363837530326_htanjloss_dewarpWc_scratch30-60knoaug_l1_best_model.pkl --bm_resume ./checkpoints-bm/dnet_dewarpnet_52_0.000705995364115_0.000154406980559_dewarpbmmeshplit_rescaled_retrain_ssimreconssiml1_best_model.pkl --visdom