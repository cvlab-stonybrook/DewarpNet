# train dewarpnet in an end to end fashion
# activations on wc,bm regression is hardtanh
# initial input is RGB image

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

from src.models import get_model
from src.loader import get_loader, get_data_path
from src.metrics import runningScore
from src.loss import *
from src.augmentations import *
import recon_loss 
import pytorch_ssim
import joint_loss
import grad_loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])

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
    out_arr=np.full((7,3,args.bm_rows,args.bm_cols),0.0)
    label_arr=np.full((7,3,args.bm_rows,args.bm_cols),0.0)
    choices=random.sample(range(n), min(n,7))
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


def write_log_file(experiment_name,losses, epoch, lrate, phase):
    with open('./checkpoints-e2e-swat3d/'+experiment_name+'.txt','a') as f:
        f.write("\n{} LRate: {} Epoch: {} wcLoss: {:.6f} wcMSE: {:.6f} wcFG: {:.6f} wcBG: {:.6f} wcGradLoss: {:.6f} bmLoss: {:.6f} bmMSE: {:.6f} UnwarpL2: {:.6f} UnwarpSSIMloss: {:.6f}".format(
                                phase, lrate, epoch, losses[0], losses[1], losses[2], losses[3],losses[4], 
                                losses[5], losses[6], losses[7], losses[8]))

def train(args):

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset+'e2e')
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), bm_size=(args.bm_rows, args.bm_cols), augmentations=True, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='valbmswat3d', img_size=(args.img_rows, args.img_cols), bm_size=(args.bm_rows, args.bm_cols), img_norm=args.img_norm)

    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)


    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        # train_labels1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Train label 1x', caption='In progress..'))
        # train_labels1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Train label 1y', caption='In progress..'))
        # train_out1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Train Output 1x', caption='In progress..'))
        # train_out1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Train Output 1y', caption='In progress..'))

        # val_labels1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Val labels 1x', caption='In progress..'))
        # val_labels1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Val labels 1y', caption='In progress..'))
        # val_out1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Val Outputs 1x', caption='In progress..'))
        # val_out1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Val Outputs 1y', caption='In progress..'))
    

        train_inputs_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train Inputs', caption='Wc_noaug'))
        train_labelswc_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train labels', caption='Wc_noaug'))
        train_outwc_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train Outputs', caption='Wc_noaug')) 
        train_labelsbm_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train labels', caption='Train GT Dewarp'))
        train_outbm_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train Outputs', caption='Train Pred Dewarp'))       
        
        val_inputs_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Inputs', caption='Wc_noaug'))
        val_labelswc_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Labels', caption='Wc_noaug'))
        val_outwc_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Outputs', caption='Wc_noaug'))
        val_labelsbm_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Labels', caption='Val GT Dewarp'))
        val_outbm_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Outputs', caption='Val Pred Dewarp'))
        

    # Setup Model
    wc_model = get_model(args.wc_arch, n_classes=3, in_channels=3)
    bm_model = get_model(args.bm_arch, n_classes=2, in_channels=3)
    
    wc_model = torch.nn.DataParallel(wc_model, device_ids=range(torch.cuda.device_count()))
    wc_model.cuda()

    bm_model = torch.nn.DataParallel(bm_model, device_ids=range(torch.cuda.device_count()))
    bm_model.cuda()
    
    # Setup Optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    optimizer= torch.optim.Adam([{'params':wc_model.parameters()},
                                 {'params':bm_model.parameters()}  ],lr=args.l_rate, weight_decay=5e-4, amsgrad=True)

    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


    # Define Loss
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()
    reconst_loss= recon_loss.Unwarploss()
    gloss= grad_loss.Gradloss(window_size=5,padding=2)
    jloss_fn=joint_loss.JointLoss()

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

    #Log file:
    trainstartwc=args.wc_resume.split('/')[-1].split('_') #experiment name split on '_', take pos 2,3: epoch & valmse
    trainstartwc=trainstartwc[2]+'x'+trainstartwc[3]
    trainstartbm=args.bm_resume.split('/')[-1].split('_') #experiment name split on '_', take pos 2,3: epoch & valmse
    trainstartbm=trainstartbm[2]+'x'+trainstartbm[3]

    experiment_name='unetnc_dnetccnl_swat3dbmFixedvYvXvXYRanChess_l1grad0.7_l1recon10.5_bghs_'+trainstartwc+'_'+trainstartbm #netwc_netbm_dataset_losswc_lossbm_augmentations_trainstartwc_trainstartbm
    log_file_name='./checkpoints-e2e-swat3d/'+experiment_name+'.txt'
    if os.path.isfile(log_file_name):
        log_file=open(log_file_name,'a')
    else:
        log_file=open(log_file_name,'w+')

    log_file.write('\n\n---------------  '+experiment_name+'  ---------------\n')
    log_file.write('\n---------------  wc start'+args.wc_resume+'  ---------------')
    log_file.write('\n---------------  bm start'+args.bm_resume+'  ---------------\n\n')
    log_file.close()

    best_valwc_mse =999999.0
    best_valbm_mse =999999.0
    

    log_count=0
    for epoch in range(epoch_start,args.n_epoch):
        avg_loss=0.0
        avg_wcloss=0.0
        avgwcl1loss=0.0
        avg_fg=0.0
        avg_bg=0.0
        avg_gloss=0.0
        train_wcmse=0.0
        avg_bmloss=0.0
        avgbml1loss=0.0
        avgrloss=0.0
        avgssimloss=0.0
        train_bmmse=0.0
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
            wc_outputs = F.interpolate(wc_outputs, size=(args.img_rows, args.img_cols), mode='bilinear')
            bm_inputs = F.interpolate(wc_outputs, size=(args.bm_rows, args.bm_cols), mode='bilinear')
            # wc_outputs_nhwc = wc_outputs.transpose(1, 2).transpose(2, 3)
            #print(target_nhwc.shape)
            bm_inputs=htan(bm_inputs)
            wc_pred_nchw=htan(wc_outputs)
            wc_l1loss = loss_fn(wc_pred_nchw, wc_labels)
            wc_gloss=gloss(wc_pred_nchw, wc_labels)
            wc_fgloss,wc_bgloss=jloss_fn(wc_pred_nchw, wc_labels)
            wc_mse=MSE(wc_pred_nchw, wc_labels)
            wc_loss=wc_l1loss+(0.7*wc_gloss)

            #track wc losses
            avg_wcloss+=float(wc_loss)
            avgwcl1loss+=float(wc_l1loss)
            avg_fg+=float(wc_fgloss)
            avg_bg+=float(wc_bgloss)
            avg_gloss+=float(wc_gloss)
            train_wcmse+=float(wc_mse)

            bm_outputs = bm_model(bm_inputs)
            bm_outputs_nhwc = bm_outputs.transpose(1, 2).transpose(2, 3)
            bm_labels_nchw=bm_labels.transpose(3,2).transpose(2,1)
            bm_l1loss = loss_fn(bm_outputs_nhwc, bm_labels)
            rloss,ssim,uworg,uwpred = reconst_loss(alb_labels,bm_outputs_nhwc,bm_labels)
            bm_mse=MSE(bm_outputs_nhwc, bm_labels)
            alb_labels_nchw=alb_labels.transpose(3, 2).transpose(2, 1)
            bm_loss=(10.0*bm_l1loss)+(0.5*rloss)
            # show_uloss(uwpred,uworg,images[:,:3,:,:])

            #track bm losses
            avg_bmloss+=float(bm_loss)
            avgbml1loss+=float(bm_l1loss)        
            avgrloss+=float(rloss)
            avgssimloss+=float(ssim)
            train_bmmse+=float(bm_mse)

            loss=wc_loss+bm_loss
            avg_loss+=float(loss)

            loss.backward()
            # for param in model.parameters():
            #     print(param.grad.data.sum())
            optimizer.step()

            if (i+1) % 50 == 0:
                avg_loss=avg_loss/50
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss))
                # print("L1:%4f, SNL1:%.4f" %(avgl1loss.item()/50,avg_nl1.item()/50))
                avg_loss=0.0
            
            if args.visdom:
                # choices=random.sample(range(images.shape[0]), 1)
                # #show batch output and labels
                # outx_arr=np.full((args.img_rows,args.img_cols),0)
                # outy_arr=np.full((args.img_rows,args.img_cols),0)
                # labelx_arr=np.full((args.img_rows,args.img_cols),0) 
                # labely_arr=np.full((args.img_rows,args.img_cols),0) 
                # idx=0
                # target_cpu=bm_outputs.detach().cpu().numpy()
                # labels_cpu=bm_labels.detach().cpu().numpy()
                # for c in choices:
                #     # labels_nchw=labels.transpose(3,2).transpose(2,1)
                #     # print(labels_nchw.shape)
                #     outx_arr=target_cpu[c,0,:,:]
                #     outy_arr=target_cpu[c,1,:,:]
                #     labelx_arr=labels_cpu[c,:,:,0]
                #     labely_arr=labels_cpu[c,:,:,1]
                #     # print(np.max(labelx_arr))
                #     # print(np.min(labelx_arr))
                #     idx+=1
                # vis.heatmap(outx_arr,
                #            win=train_out1x_win,
                #            opts=dict(title='Train Output 1x', caption='In progress..'))
                # vis.heatmap(outy_arr,
                #            win=train_out1y_win,
                #            opts=dict(title='Train Output 1y', caption='In progress..'))
                # vis.heatmap(labelx_arr,
                #            win=train_labels1x_win,
                #            opts=dict(title='Train Label 1x', caption='In progress..'))
                # vis.heatmap(labely_arr,
                #            win=train_labels1y_win,
                #            opts=dict(title='Train Label 1y', caption='In progress..'))
                labelopts=dict(title='Train Label', caption='Gt unwarp_'+experiment_name)
                outopts=dict(title='Train Out', caption='Pred. unwarp_'+experiment_name)
                show_uloss_visdom(vis,uwpred,uworg,train_labelsbm_win,train_outbm_win,labelopts,outopts,args)
                choices=random.sample(range(images.shape[0]), min(images.shape[0],5))
                #show batch output and labels
                out_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)
                inp_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)
                label_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)  
                
                target_cpu=wc_pred_nchw.detach().cpu().numpy()
                labels_cpu=wc_labels.detach().cpu().numpy()
                inp_cpu=images.detach().cpu().numpy()
                idx=0
                # f,axarr=plt.subplots(5,3)
                for c in choices:
                    # print(labels_nchw.shape)
                    out_arr[idx,:,:,:]=target_cpu[c]
                    label_arr[idx,:,:,:]=labels_cpu[c]
                    inp_arr[idx,0,:,:]=inp_cpu[c][0,:,:]
                    inp_arr[idx,1,:,:]=inp_cpu[c][1,:,:]
                    inp_arr[idx,2,:,:]=inp_cpu[c][2,:,:]
                    # axarr[idx][0].imshow(out_arr[idx].transpose(1,2,0))
                    # axarr[idx][1].imshow(label_arr[idx].transpose(1,2,0))
                    # axarr[idx][2].imshow(inp_arr[idx].transpose(1,2,0)[:,:,::-1])
                    idx+=1
                vis.images(out_arr,
                           win=train_outwc_win,
                           opts=dict(title='Train Outputs', caption='WcvarX_'+experiment_name))
                vis.images(label_arr,
                           win=train_labelswc_win,
                           opts=dict(title='Train Labels', caption='WcvarX_'+experiment_name))
                vis.images(inp_arr,
                           win=train_inputs_win,
                           opts=dict(title='Train Inputs', caption='WcvarX_'+experiment_name))


        # print("L1:%4f, SNL1:%.4f" %(avgl1loss.item()/len(trainloader),avg_nl1.item()/len(trainloader)))

        train_wcmse=train_wcmse/len(trainloader)
        train_bmmse=train_bmmse/len(trainloader)
        print("Training wcMSE:'{}'".format(train_wcmse))
        print("Training bmMSE:'{}'".format(train_bmmse))

        train_losses=[avgwcl1loss/len(trainloader), train_wcmse, avg_fg/len(trainloader),avg_bg/len(trainloader),avg_gloss/len(trainloader),
                      avgbml1loss/len(trainloader), train_bmmse, avgrloss/len(trainloader),avgssimloss/len(trainloader)]
        
        lrate=get_lr(optimizer)
        write_log_file(experiment_name, train_losses, epoch+1, lrate, 'Train')


        
        wc_model.eval()
        bm_model.eval()
        wc_val_l1=0.0
        wc_val_mse=0.0
        wc_val_fg=0.0
        wc_val_bg=0.0
        wc_val_gloss=0.0
        bm_val_l1=0.0
        bm_val_mse=0.0
        bm_val_rloss=0.0
        bm_val_ssim=0.0
        for i_val, (imgs_val, wcs_val, bms_val, albs_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(imgs_val.cuda())
                wc_labels_val = Variable(wcs_val.cuda())
                bm_labels_val = Variable(bms_val.cuda())    #this is nhwc
                alb_labels_val =Variable(albs_val.cuda())

                wc_outputs_val = wc_model(images_val)       #this is (nchw)  
                wc_outputs_val = F.interpolate(wc_outputs_val, size=(args.img_rows, args.img_cols), mode='bilinear')
                bm_inputs_val = F.interpolate(wc_outputs_val, size=(args.bm_rows, args.bm_cols), mode='bilinear')
                # wc_outputs_nhwc = wc_outputs.transpose(1, 2).transpose(2, 3)
                #print(target_nhwc.shape)
                wc_pred_nchw_val=htan(wc_outputs_val)
                bm_inputs_val=htan(bm_inputs_val)
                wc_l1 = loss_fn(wc_pred_nchw_val, wc_labels_val)
                wc_mse = MSE(wc_pred_nchw_val, wc_labels_val)
                wc_gloss = gloss(wc_pred_nchw_val, wc_labels_val)
                wc_fg, wc_bg = jloss_fn(wc_pred_nchw_val, wc_labels_val)

                bm_outputs_val = bm_model(bm_inputs_val)
                bm_outputs_nhwc_val = bm_outputs_val.transpose(1, 2).transpose(2, 3)
                bm_labels_nchw_val=bm_labels_val.transpose(3,2).transpose(2,1)
                
                bm_l1 = loss_fn(bm_outputs_nhwc_val, bm_labels_val)
                rloss,ssim,uworg,uwpred = reconst_loss(alb_labels_val,bm_outputs_nhwc_val,bm_labels_val)
                bm_mse= MSE(bm_outputs_nhwc_val, bm_labels_val)
                alb_labels_nchw_val=alb_labels_val.transpose(3, 2).transpose(2, 1)

                wc_val_l1+=float(wc_l1.cpu())
                wc_val_mse+=float(wc_mse.cpu())
                wc_val_gloss+=float(wc_gloss.cpu())
                wc_val_fg+=float(wc_fg.cpu())
                wc_val_bg+=float(wc_bg.cpu())
                
                bm_val_l1+=float(bm_l1.cpu())
                bm_val_rloss+=float(rloss.cpu())
                bm_val_ssim+=float(ssim.cpu())
                bm_val_mse+=float(bm_mse.cpu())

            if args.visdom:
                # choices=random.sample(range(images.shape[0]), 1)
                # #show batch output and labels
                # outx_arr=np.full((args.img_rows,args.img_cols),0)
                # outy_arr=np.full((args.img_rows,args.img_cols),0)
                # labelx_arr=np.full((args.img_rows,args.img_cols),0) 
                # labely_arr=np.full((args.img_rows,args.img_cols),0) 
                # idx=0
                # for c in choices:
                #     # labels_nchw=labels_val.transpose(3,2).transpose(2,1)
                #     # print(labels_nchw.shape)
                #     target_cpu=bm_outputs_val.detach().cpu().numpy()
                #     labels_cpu=bm_labels_val.detach().cpu().numpy()
                #     outx_arr=target_cpu[c,0,:,:]
                #     outy_arr=target_cpu[c,1,:,:]
                #     labelx_arr=labels_cpu[c,:,:,0]
                #     labely_arr=labels_cpu[c,:,:,1]
                #     idx+=1
                # vis.heatmap(outx_arr,
                #            win=val_out1x_win,
                #            opts=dict(title='Val Output 1x', caption='In progress..'))
                # vis.heatmap(outy_arr,
                #            win=val_out1y_win,
                #            opts=dict(title='Val Output 1y', caption='In progress..'))
                # vis.heatmap(labelx_arr,
                #            win=val_labels1x_win,
                #            opts=dict(title='Val Label 1x', caption='In progress..'))
                # vis.heatmap(labely_arr,
                #            win=val_labels1y_win,
                #            opts=dict(title='Val Label 1y', caption='In progress..'))
                
                labelopts=dict(title='Val Label', caption='Gt unwarp_'+experiment_name)
                outopts=dict(title='Val Out', caption='Pred. unwarp_'+experiment_name)
                show_uloss_visdom(vis,uwpred,uworg,val_labelsbm_win,val_outbm_win,labelopts,outopts,args)

                choices=random.sample(range(images_val.shape[0]), min(5,images_val.shape[0]))
                #show batch output and labels
                out_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)
                inp_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)
                label_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)  
                
                target_cpu=wc_pred_nchw_val.detach().cpu().numpy()
                labels_cpu=wc_labels_val.detach().cpu().numpy()
                inp_cpu=images_val.detach().cpu().numpy()
                idx=0
                for c in choices:
                    # print(labels_nchw.shape)
                    out_arr[idx,:,:,:]=target_cpu[c]
                    label_arr[idx,:,:,:]=labels_cpu[c]
                    inp_arr[idx,0,:,:]=inp_cpu[c][0,:,:]
                    inp_arr[idx,1,:,:]=inp_cpu[c][1,:,:]
                    inp_arr[idx,2,:,:]=inp_cpu[c][2,:,:]
                    idx+=1
                
                vis.images(out_arr,
                           win=val_outwc_win,
                           opts=dict(title='Val Outputs', caption='WcvarX_'+experiment_name))
                vis.images(label_arr,
                           win=val_labelswc_win,
                           opts=dict(title='Val Labels', caption='WcvarX_'+experiment_name))
                vis.images(inp_arr,
                           win=val_inputs_win,
                           opts=dict(title='Val Inputs', caption='WcvarX_'+experiment_name))

        wc_val_mse=wc_val_mse/len(valloader)
        bm_val_mse=bm_val_mse/len(valloader)
        print("Validation wcMSE:'{}'".format(wc_val_mse))
        print("Validation bmMSE:'{}'".format(bm_val_mse))

        val_losses=[wc_val_l1/len(valloader), wc_val_mse, wc_val_fg/len(valloader),wc_val_bg/len(valloader),wc_val_gloss/len(valloader),
                      bm_val_l1/len(valloader), bm_val_mse, bm_val_rloss/len(valloader),bm_val_ssim/len(valloader)]
        write_log_file(experiment_name, val_losses, epoch+1, lrate,'Val')

        #reduce learning rate
        sched.step(wc_val_mse)

        if wc_val_mse < best_valwc_mse:
            best_valwc_mse=wc_val_mse
            state = {'epoch': epoch+1,
                     'model_state': wc_model.state_dict(),}
            torch.save(state, "./checkpoints-e2e-swat3d/{}_{}_{}_{}_{}_{}_best_model.pkl".format(args.wc_arch, args.dataset, epoch+1,wc_val_mse,train_wcmse, experiment_name))
        if bm_val_mse < best_valbm_mse:
            best_valbm_mse=bm_val_mse
            state = {'epoch': epoch+1,
                     'model_state': bm_model.state_dict(),}
            torch.save(state, "./checkpoints-e2e-swat3d/{}_{}_{}_{}_{}_{}_best_model.pkl".format(args.bm_arch, args.dataset, epoch+1,bm_val_mse,train_bmmse, experiment_name))

        if (epoch+1) % 5 == 0:
                # best_iou = score['Mean IoU : \t']
                state = {'epoch': epoch+1,
                         'model_state': wc_model.state_dict(),}
                torch.save(state, "./checkpoints-e2e-swat3d/{}_{}_{}_{}_{}_{}_model.pkl".format(args.wc_arch, args.dataset, epoch+1,wc_val_mse,train_wcmse, experiment_name))

                state = {'epoch': epoch+1,
                         'model_state': bm_model.state_dict(),}
                torch.save(state, "./checkpoints-e2e-swat3d/{}_{}_{}_{}_{}_{}_model.pkl".format(args.bm_arch, args.dataset, epoch+1,bm_val_mse,train_bmmse, experiment_name))

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
    parser.add_argument('--bm_rows', nargs='?', type=int, default=128, 
                        help='Height of the input image')
    parser.add_argument('--bm_cols', nargs='?', type=int, default=128, 
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


 #CUDA_VISIBLE_DEVICES=1 python trainS3dE2E.py --wc_arch unetnc --bm_arch dnetccnl --dataset dewarpnet --img_norm --n_epoch 100 --batch_size 50 --l_rate 0.0001 --wc_resume  --bm_resume 