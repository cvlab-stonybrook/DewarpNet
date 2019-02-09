# code to train backward mapping regression from world coordinates (v2)
# models are in checkpoints-bm-swat3d/ 

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group['lr'])


def visualize(epoch,model,layer):    
    #get conv layers
    conv_layers=[]
    for m in model.modules():
        if isinstance(m,torch.nn.modules.conv.Conv2d):
            conv_layers.append(m)

    # print conv_layers[layer].weight.data.cpu().numpy().shape
    tensor=conv_layers[layer].weight.data.cpu()
    vistensor(tensor, epoch, ch=0, allkernels=False, nrow=8, padding=1)


def vistensor(tensor, epoch, ch=0, allkernels=False, nrow=8, padding=1): 
    '''
    vistensor: visuzlization tensor
        @ch: visualization channel 
        @allkernels: visualization all tensors
    ''' 
    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )
    # print rows
    # print tensor.shape
    grid = utils.make_grid(tensor, nrow=8, normalize=True, padding=padding)
    # print grid.shape
    plt.figure( figsize=(10,10), dpi=200 )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.savefig('./generated/filters_layer1_dwuv_'+str(epoch)+'.png')
    plt.close()


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
    out_arr=np.full((7,3,args.img_rows,args.img_cols),0.0)
    label_arr=np.full((7,3,args.img_rows,args.img_cols),0.0)
    choices=random.sample(range(n),min(n,7))
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
    with open('./checkpoints-bm-swat3d/'+experiment_name+'.txt','a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {} MSE: {} UnwarpL2: {} UnwarpSSIMloss: {}".format(phase, lrate , epoch, losses[0], losses[1], losses[2], losses[3]))


def train(args):

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset+'bmni')
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='valbmswat3d', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    #running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()
        
        train_labels_win = vis.images(np.full((7,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train labels', caption='Train GT Dewarp'))
        train_out_win = vis.images(np.full((7,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train Outputs', caption='Train Pred Dewarp'))
        val_labels_win = vis.images(np.full((7,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Labels', caption='Val GT Dewarp'))
        val_out_win = vis.images(np.full((7,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Outputs', caption='Val Pred Dewarp'))
        # val_labels1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Val labels 1x', caption='In progress..'))
        # val_labels1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Val labels 1y', caption='In progress..'))
        # val_out1x_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Val Outputs 1x', caption='In progress..'))
        # val_out1y_win = vis.heatmap(np.full((args.img_rows, args.img_cols),0),
        #                            opts=dict(title='Val Outputs 1y', caption='In progress..'))
        

    # Setup Model
    model = get_model(args.arch, n_classes,in_channels=3)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
        optimizer= torch.optim.Adam(model.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)

    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, verbose=True)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        MSE = nn.MSELoss()
        loss_fn = nn.L1Loss()
        reconst_loss= recon_loss.Unwarploss()
        # ssim_loss=pytorch_ssim.SSIM(window_size=10,channels=2)
    epoch_start=0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            pretrained_dict=checkpoint['model_state']
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            # model.load_state_dict(checkpoint['model_state'])
            # optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    #Log file:
    experiment_name='dnetccnl_htan_swat3dbmFixedvYvXvXYRanKChess48_l1recon10.5_noaug_retrainFixedvYvXvXYRanChess48l1recon10.5' #network_activation(t=[0,1])_dataset_lossparams_augmentations_trainstart
    log_file_name='./checkpoints-bm-swat3d/'+experiment_name+'.txt'
    if os.path.isfile(log_file_name):
        log_file=open(log_file_name,'a')
    else:
        log_file=open(log_file_name,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    best_val_uwarpssim = 99999.0
    best_val_mse=99999.0
    log_count=0
    for epoch in range(epoch_start,args.n_epoch):
        avg_loss=0.0
        avgl1loss=0.0
        avgrloss=0.0
        avgssimloss=0.0
        train_mse=0.0
        model.train()
        # save filter visualization
        # visualize(epoch,model,layer=0)
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            # print torch.max(labels)
            # print torch.min(labels)

            optimizer.zero_grad()
            # print (images.shape)
            target = model(images[:,3:,:,:])
            target_nhwc = target.transpose(1, 2).transpose(2, 3)
            labels_nchw=labels.transpose(3,2).transpose(2,1)
            l1loss = loss_fn(target_nhwc, labels)
            # ssloss=1-ssim_loss(target,labels_nchw)
            rloss,ssim,uworg,uwpred = reconst_loss(images,target_nhwc,labels)
            labels_nchw=labels.transpose(3, 2).transpose(2, 1)
            loss=(10.0*l1loss)+(0.5*rloss) #+ (0.3*ssim)
            # show_uloss(uwpred,uworg,images[:,:3,:,:])
            # loss=l1loss  
            avgl1loss+=float(l1loss)        
            avg_loss+=float(loss)
            avgrloss+=float(rloss)
            avgssimloss+=float(ssim)
            
            train_mse+=MSE(target_nhwc, labels).item()

            loss.backward()
            # for param in model.parameters():
            #     print(param.grad.data.sum())
            optimizer.step()

            if (i+1) % 50 == 0:
                avg_loss=avg_loss/50
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss))
                avg_loss=0.0

            if args.visdom:
                # choices=random.sample(range(images.shape[0]), 1)
                #show batch output and labels
                # outx_arr=np.full((args.img_rows,args.img_cols),0)
                # outy_arr=np.full((args.img_rows,args.img_cols),0)
                # labelx_arr=np.full((args.img_rows,args.img_cols),0) 
                # labely_arr=np.full((args.img_rows,args.img_cols),0) 
                # idx=0
                # target_cpu=target.detach().cpu().numpy()
                # labels_cpu=labels.detach().cpu().numpy()
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
                show_uloss_visdom(vis,uwpred,uworg,train_labels_win,train_out_win,labelopts,outopts,args)


        # print("L1:%4f, SNL1:%.4f" %(avgl1loss.item()/len(trainloader),avg_nl1.item()/len(trainloader)))

        print("Training L1:%4f" %(avgl1loss/len(trainloader)))
        train_mse=train_mse/len(trainloader)
        print("Training mse:'{}'".format(train_mse))
        avgssimloss=avgssimloss/len(trainloader)
        train_losses=[avgl1loss/len(trainloader), train_mse, avgrloss/len(trainloader),avgssimloss ]
        lrate=get_lr(optimizer)
        write_log_file(experiment_name, train_losses,epoch+1, lrate ,'Train')

        model.eval()
        val_loss=0.0
        val_l1loss=0.0
        val_mse=0.0
        val_rloss=0.0
        val_ssimloss=0.0

        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(images_val.cuda())
                labels_val = Variable(labels_val.cuda())

                target = model(images_val[:,3:,:,:])
                target_nhwc = target.transpose(1, 2).transpose(2, 3)
                labels_nchw=labels_val.transpose(3,2).transpose(2,1)
                pred=target_nhwc.data.cpu()
                gt = labels_val.cpu()
                # loss = loss_fn(pred, gt)
                l1loss = loss_fn(target_nhwc, labels_val)
                # ssloss=1-ssim_loss(target,labels_nchw)
                rloss,ssim,uworg,uwpred = reconst_loss(images_val,target_nhwc,labels_val)
                #loss=(0.9*loss)+ (0.05*rloss.cpu()) +(0.05*ssim.cpu())
                # loss=(0.4*l1loss.cpu())+(0.6*rloss.cpu()) #+ (0.3*ssim)
                val_l1loss+=float(l1loss.cpu())
                # val_loss+=float(loss)
                val_rloss+=float(rloss.cpu())
                val_ssimloss+=float(ssim.cpu())
                val_mse+=float(MSE(pred, gt))
            if args.visdom:
                # choices=random.sample(range(images_val.shape[0]), 1)
                # #show batch output and labels
                # outx_arr=np.full((args.img_rows,args.img_cols),0)
                # outy_arr=np.full((args.img_rows,args.img_cols),0)
                # labelx_arr=np.full((args.img_rows,args.img_cols),0) 
                # labely_arr=np.full((args.img_rows,args.img_cols),0) 
                # idx=0
                # for c in choices:
                #     labels_nchw=labels_val.transpose(3,2).transpose(2,1)
                #     # print(labels_nchw.shape)
                #     target_cpu=target.detach().cpu().numpy()
                #     labels_cpu=labels_val.detach().cpu().numpy()
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
                show_uloss_visdom(vis,uwpred,uworg,val_labels_win,val_out_win,labelopts,outopts,args)



        print("val loss at epoch {}:: {}".format(epoch+1,val_l1loss/len(valloader)))
        val_mse=val_mse/len(valloader)
        print("val mse: {}".format(val_mse)) 
        val_ssimloss=val_ssimloss/len(valloader)

        val_losses=[val_l1loss/len(valloader), val_mse, val_rloss/len(valloader), val_ssimloss]
        write_log_file(experiment_name, val_losses, epoch+1, lrate, 'Val')

        #reduce learning rate
        sched.step(val_mse) 

        if val_mse < best_val_mse:
            best_val_mse=val_mse
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "./checkpoints-bm-swat3d/{}_{}_{}_{}_{}_{}_best_model.pkl".format(args.arch, args.dataset, epoch+1,val_mse,train_mse,experiment_name))

        if (epoch+1) % 10 == 0:
            # best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "./checkpoints-bm-swat3d/{}_{}_{}_{}_{}_{}_model.pkl".format(args.arch, args.dataset, epoch+1,val_mse,train_mse,experiment_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
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
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    parser.add_argument('--visdom', dest='visdom', action='store_true', 
                        help='Enable visualization(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false', 
                        help='Disable visualization(s) on visdom | False by default')
    parser.set_defaults(visdom=False)

    args = parser.parse_args()
    train(args)



#CUDA_VISIBLE_DEVICES=1 python trainS3dbmnoimgv2.py --arch dnetnbn --dataset swat3d --img_rows 128 --img_cols 128 --img_norm --n_epoch 250 --batch_size 50 --l_rate 0.0001 --visdom
