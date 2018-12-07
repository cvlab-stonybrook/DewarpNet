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



def train(args):

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset+'bmuv')
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='valWcdewarp', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=16, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=16)

    # Setup Metrics
    #running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        train_loss_window = vis.line(X=torch.zeros((1)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))
        val_loss_window = vis.line(X=torch.zeros((1)).cpu(),
                   Y=torch.zeros((1)).cpu(),
                   opts=dict(xlabel='minibatches',
                             ylabel='Loss',
                             title='Validation Loss',
                             legend=['Loss']))

    # Setup Model
    model = get_model(args.arch, n_classes,in_channels=9)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        #loss_fn = cross_entropy2d
        MSE = nn.MSELoss()
        loss_fn = nn.L1Loss()
        #ssim_loss=pytorch_ssim.SSIM(window_size=20)
        # nloss= norm_loss__.NORMloss(window_size=5)
    epoch_start=0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
            epoch_start=checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    best_val_mse = 999999.0
    log_count=0
    for epoch in range(epoch_start,args.n_epoch):
        avg_loss=0.0
        avgl1loss=0.0
        avg_nl1=0.0
        train_mse=0.0
        model.train()
        # save filter visualization
        visualize(epoch,model,layer=0)
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            # print torch.max(labels)
            # print torch.min(labels)

            optimizer.zero_grad()
            # print (images.shape)
            target = model(images)
            target_nhwc = target.transpose(1, 2).transpose(2, 3)
            loss = loss_fn(target_nhwc, labels)
            
            avg_loss+=loss
            
            train_mse+=MSE(target_nhwc, labels).item()

            loss.backward()
            # for param in model.parameters():
            #     print(param.grad.data.sum())
            optimizer.step()

            if (i+1) % 50 == 0:
                avg_loss=avg_loss/50
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss))
                # print("L1:%4f, SNL1:%.4f" %(avgl1loss.item()/50,avg_nl1.item()/50))
                if args.visdom:
                    vis.line(
                        X=torch.ones((1)).cpu() * epoch,
                        Y=torch.Tensor([avg_loss]).cpu(),
                        win=train_loss_window,
                        update='append')
                    log_count+=1
                    avg_loss=0.0

        # print("L1:%4f, SNL1:%.4f" %(avgl1loss.item()/len(trainloader),avg_nl1.item()/len(trainloader)))

        train_mse=train_mse/len(trainloader)
        print("Training mse:'{}'".format(train_mse))

        model.eval()
        val_loss=0.0
        val_mse=0.0
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(images_val.cuda())
                labels_val = Variable(labels_val.cuda())

                target = model(images_val)
                target_nhwc = target.transpose(1, 2).transpose(2, 3)
                pred=target_nhwc.data.cpu()
                gt = labels_val.cpu()
                loss = loss_fn(pred, gt)
                val_loss+=loss
                val_mse+=MSE(pred, gt)
            if args.visdom:
                vis.line(
                    X=torch.ones((1)).cpu() * i,
                    Y=torch.Tensor([loss.item()]).cpu(),
                    win=val_loss_window,
                    update='append')
        #     running_metrics.update(gt, pred)

        # score, class_iou = running_metrics.get_scores()
        # for k, v in score.items():
        #     print(k, v)
        # running_metrics.reset()

        print("val loss at epoch {}:: {}".format(epoch+1,val_loss/len(valloader)))
        val_mse=val_mse/len(valloader)
        print("val mse: {}".format(val_mse))
        if args.visdom:
            vis.line(
                X=torch.ones((1)).cpu() * epoch,
                Y=torch.Tensor([val_mse]).cpu(),
                win=val_loss_window,
                update='append')
        
        if val_mse < best_val_mse:
            best_val_mse=val_mse
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "./checkpoints-bmuv/{}_{}_{}_{}_{}_dewarpbmuv_scratch_l1_best_model.pkl".format(args.arch, args.dataset, epoch+1,val_mse,train_mse))

        if (epoch+1) % 10 == 0:
            # best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "./checkpoints-bmuv/{}_{}_{}_{}_{}_dewarpbmuv_scratch_l1_model.pkl".format(args.arch, args.dataset, epoch+1,val_mse,train_mse))

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
