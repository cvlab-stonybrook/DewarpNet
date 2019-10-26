# code to train backward mapping regression from GT world coordinates
# models are saved in checkpoints-bm/ 

import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils import data
from torchvision import utils
from tqdm import tqdm

from models import get_model
from loaders import get_loader
from utils import show_unwarp_tnsboard,  get_lr
import recon_lossc
import pytorch_ssim



def write_log_file(log_file_name,losses, epoch, lrate, phase):
    with open(log_file_name,'a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {} MSE: {} UnwarpL2: {} UnwarpSSIMloss: {}".format(phase, lrate, epoch, losses[0], losses[1], losses[2], losses[3]))

def train(args):

    # Setup Dataloader
    data_loader = get_loader('doc3dbmnic')
    data_path = args.data_path
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols))
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Model
    model = get_model(args.arch, n_classes,in_channels=3)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Optimizer
    optimizer= torch.optim.Adam(model.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)

    # LR Scheduler
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Losses
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()
    reconst_loss= recon_lossc.Unwarploss()

    epoch_start=0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            # optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    # Log file:
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    experiment_name='dnetccnl_htan_swat3dmini1kbm_l1_noaug_scratch' #network_activation(t=[-1,1])_dataset_lossparams_augmentations_trainstart
    log_file_name=os.path.join(args.logdir,experiment_name+'.txt')
    if os.path.isfile(log_file_name):
        log_file=open(log_file_name,'a')
    else:
        log_file=open(log_file_name,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    # Setup tensorboard for visualization
    if args.tboard:
        # save logs in runs/<experiment_name> 
        writer = SummaryWriter(comment=experiment_name)

    best_val_uwarpssim = 99999.0
    best_val_mse=99999.0
    global_step=0

    for epoch in range(epoch_start,args.n_epoch):
        avg_loss=0.0
        avgl1loss=0.0
        avgrloss=0.0
        avgssimloss=0.0
        train_mse=0.0
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            target = model(images[:,3:,:,:])
            target_nhwc = target.transpose(1, 2).transpose(2, 3)
            l1loss = loss_fn(target_nhwc, labels)
            rloss,ssim,uworg,uwpred = reconst_loss(images[:,:-1,:,:],target_nhwc,labels)
            loss=(10.0*l1loss) +(0.5*rloss) #+ (0.3*ssim)
            # loss=l1loss  
            avgl1loss+=float(l1loss)        
            avg_loss+=float(loss)
            avgrloss+=float(rloss)
            avgssimloss+=float(ssim)
            
            train_mse+=MSE(target_nhwc, labels).item()

            loss.backward()
            optimizer.step()
            global_step+=1

            if (i+1) % 50 == 0:
                avg_loss=avg_loss/50
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss))
                avg_loss=0.0

            if args.tboard and  (i+1) % 20 == 0:
                show_unwarp_tnsboard(global_step, writer,uwpred,uworg,8,'Train GT unwarp', 'Train Pred Unwarp')
                writer.add_scalar('BM: L1 Loss/train', avgl1loss/(i+1), global_step)
                writer.add_scalar('CB: Recon Loss/train', avgrloss/(i+1), global_step)
                writer.add_scalar('CB: SSIM Loss/train', avgssimloss/(i+1), global_step)


        avgssimloss=avgssimloss/len(trainloader)
        avgrloss=avgrloss/len(trainloader)
        avgl1loss=avgl1loss/len(trainloader)
        train_mse=train_mse/len(trainloader)
        print("Training L1:%4f" %(avgl1loss))
        print("Training MSE:'{}'".format(train_mse))
        train_losses=[avgl1loss, train_mse, avgrloss ,avgssimloss ]
        lrate=get_lr(optimizer)
        write_log_file(log_file_name, train_losses,epoch+1, lrate,'Train')
        
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
                pred=target_nhwc.data.cpu()
                gt = labels_val.cpu()
                l1loss = loss_fn(target_nhwc, labels_val)
                rloss,ssim,uworg,uwpred = reconst_loss(images_val[:,:-1,:,:],target_nhwc,labels_val)
                val_l1loss+=float(l1loss.cpu())
                val_rloss+=float(rloss.cpu())
                val_ssimloss+=float(ssim.cpu())
                val_mse+=float(MSE(pred, gt))
            if args.tboard:
                show_unwarp_tnsboard(epoch+1, writer,uwpred,uworg,8,'Val GT unwarp', 'Val Pred Unwarp')

        val_l1loss=val_l1loss/len(valloader)
        val_mse=val_mse/len(valloader)
        val_ssimloss=val_ssimloss/len(valloader)
        val_rloss= val_rloss/len(valloader)
        print("val loss at epoch {}:: {}".format(epoch+1,val_l1loss))
        print("val mse: {}".format(val_mse)) 
        val_losses=[val_l1loss, val_mse, val_rloss , val_ssimloss]
        write_log_file(log_file_name, val_losses, epoch+1, lrate, 'Val')
        if args.tboard:
            # log the val losses
            writer.add_scalar('BM: L1 Loss/val', val_l1loss, epoch+1)
            writer.add_scalar('CB: Recon Loss/val', val_rloss, epoch+1)
            writer.add_scalar('CB: SSIM Loss/val', val_ssimloss, epoch+1)

        #reduce learning rate
        sched.step(val_mse) 

        if val_mse < best_val_mse:
            best_val_mse=val_mse
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, args.logdir+"{}_{}_{}_{}_{}_best_model.pkl".format(args.arch, epoch+1,val_mse,train_mse,experiment_name))

        if (epoch+1) % 10 == 0:
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, args.logdir+"{}_{}_{}_{}_{}_model.pkl".format(args.arch, epoch+1,val_mse,train_mse,experiment_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='dnetccnl', 
                        help='Architecture to use [\'dnetccnl, unetnc\']')
    parser.add_argument('--data_path', nargs='?', type=str, default='', 
                        help='Data path to load data')
    parser.add_argument('--img_rows', nargs='?', type=int, default=128, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=128, 
                        help='Width of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--logdir', nargs='?', type=str, default='./checkpoints-bm/',    
                        help='Path to store the loss logs')
    parser.add_argument('--tboard', dest='tboard', action='store_true', 
                        help='Enable visualization(s) on tensorboard | False by default')
    parser.set_defaults(tboard=False)

    args = parser.parse_args()
    train(args)



#CUDA_VISIBLE_DEVICES=1 python trainS3dbmnoimg.py --arch dnetccnl --dataset swat3d --img_rows 128 --img_cols 128 --img_norm --n_epoch 250 --batch_size 50 --l_rate 0.0001 --tboard --data_path /media/hilab/sagniksSSD/Sagnik/DewarpNet