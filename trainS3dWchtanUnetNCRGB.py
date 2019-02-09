# code to train world coord regression from RGB Image
# models are saved in checkpoints-swat3d/ 

import sys, os
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.init as init
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
import joint_loss
import grad_loss

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
    plt.savefig('./generated/filters_layer1_s3wc_'+str(epoch)+'.png')
    plt.close()


def init_weights(m):
    if type(m) == nn.Conv2d:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias,0.0)


def write_log_file(experiment_name,losses, epoch, lrate, phase):
    with open('./checkpoints-swat3d/'+experiment_name+'.txt','a') as f:
        f.write("\n{} LRate: {} Epoch: {} Loss: {} MSE: {} FG: {} BG: {} GradLoss: {}".format(phase, lrate, epoch, losses[0], losses[1], losses[2], losses[3], losses[4]))


def train(args):

    # Setup Augmentations
    # data_aug= Compose([RandomRotate(10),                                        
    #                    RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset+'wc')
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm,augmentations=True)
    v_loader = data_loader(data_path, is_transform=True, split='valswat3d', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    #running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()
        train_inputs_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train Inputs', caption='Wc_noaug'))
        train_labels_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train labels', caption='Wc_noaug'))
        train_out_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Train Outputs', caption='Wc_noaug'))
        
        val_inputs_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Inputs', caption='Wc_noaug'))
        val_labels_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Labels', caption='Wc_noaug'))
        val_out_win = vis.images(np.full((5,3, args.img_rows, args.img_cols),0),
                                   opts=dict(title='Val Outputs', caption='Wc_noaug'))

    # Setup Model
    model = get_model(args.arch, n_classes,in_channels=3)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    # sigm=nn.Sigmoid()
    # softp=nn.Softplus(1.0,1.0)
    htan = nn.Hardtanh(0,1.0)
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
        optimizer= torch.optim.Adam(model.parameters(),lr=args.l_rate, weight_decay=5e-4, amsgrad=True)

    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        #loss_fn = cross_entropy2d
        MSE = nn.MSELoss()
        loss_fn = nn.L1Loss()
        jloss_fn=joint_loss.JointLoss()
        #ssim_loss=pytorch_ssim.SSIM(window_size=20)
        gloss= grad_loss.Gradloss(window_size=5,padding=2)
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
    
    #Log file:
    experiment_name='htan_swat3dWcRGBvarYvarXvarXYRanKNcrop_l1grad_bghsaugk_varYvarXvarXYRanKNl1grad' #activation_dataset_lossparams_augmentations_trainstart
    log_file_name='./checkpoints-swat3d/'+experiment_name+'.txt'
    if os.path.isfile(log_file_name):
        log_file=open(log_file_name,'a')
    else:
        log_file=open(log_file_name,'w+')

    log_file.write('\n---------------  '+experiment_name+'  ---------------\n')
    log_file.close()

    best_val_mse = 99999.0
    log_count=0
    for epoch in range(epoch_start,args.n_epoch):
        avg_loss=0.0
        avgl1loss=0.0
        avg_fg=0.0
        avg_bg=0.0
        avg_gloss=0.0
        train_mse=0.0
        model.train()
        # save filter visualization
        visualize(epoch,model,layer=0)
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)
            target = F.upsample(outputs, size=(args.img_rows, args.img_cols), mode='bilinear')
            target_nhwc = target.transpose(1, 2).transpose(2, 3)
            #print(target_nhwc.shape)
            pred_nhwc=htan(target_nhwc)
            pred_nchw=htan(target)
            labels_nchw=labels.transpose(3,2).transpose(2,1)

            #sdissim=1-ssim_loss(target_nhwc,labels_3)
            g_loss=gloss(pred_nchw, labels_nchw)
            # print(g_loss.item())
            l1loss = loss_fn(pred_nhwc, labels)
            fg_loss,bg_loss=jloss_fn(pred_nchw,labels_nchw)
            # print(g_loss.__class__)
            loss=l1loss+g_loss
            # _loss=(0.4*fg_loss) + (0.6*bg_loss)
            avgl1loss+=float(l1loss)
            # avg_nl1+=nlossl1
            avg_loss+=float(loss)
            avg_gloss+=float(g_loss)
            avg_bg+=float(bg_loss)
            avg_fg+=float(fg_loss)
            #loss = 0.5*nloss(pred, labels_nchw)
            train_mse+=float(MSE(pred_nhwc, labels).item())

            loss.backward()
            # for param in model.parameters():
            #     print(param.grad.data.sum())
            optimizer.step()

            if (i+1) % 50 == 0:
                print("Epoch[%d/%d] Batch [%d/%d] Loss: %.4f" % (epoch+1,args.n_epoch,i+1, len(trainloader), avg_loss/50.0))
                avg_loss=0.0
                # print("L1:%4f, SNL1:%.4f" %(avgl1loss.item()/50,avg_nl1.item()/50))
                if args.visdom:
                    choices=random.sample(range(images.shape[0]), 5)
                    #show batch output and labels
                    out_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)
                    inp_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)
                    label_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)  
                    
                    labels_nchw=labels.transpose(3,2).transpose(2,1)
                    target_cpu=pred_nchw.detach().cpu().numpy()
                    labels_cpu=labels_nchw.detach().cpu().numpy()
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
                    # plt.show()
                    # plt.savefig('./generated/inp_'+str(i+1)+'.png')
                    # plt.close()
                    vis.images(out_arr,
                               win=train_out_win,
                               opts=dict(title='Train Outputs', caption='WcvarX_'+experiment_name))
                    vis.images(label_arr,
                               win=train_labels_win,
                               opts=dict(title='Train Labels', caption='WcvarX_'+experiment_name))
                    vis.images(inp_arr,
                               win=train_inputs_win,
                               opts=dict(title='Train Inputs', caption='WcvarX_'+experiment_name))
            # break


        print("Training L1:%4f" %(avgl1loss/len(trainloader)))
        train_mse=train_mse/len(trainloader)
        print("Training MSE:'{}'".format(train_mse))
        train_losses=[avgl1loss/len(trainloader), train_mse, avg_fg/len(trainloader),avg_bg/len(trainloader),avg_gloss/len(trainloader)]

        lrate=get_lr(optimizer)
        write_log_file(experiment_name, train_losses,epoch+1, lrate,'Train')
        

        model.eval()
        val_loss=0.0
        val_mse=0.0
        val_bg=0.0
        val_fg=0.0
        val_gloss=0.0
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(images_val.cuda())
                labels_val = Variable(labels_val.cuda())

                outputs = model(images_val)
                #pred = outputs.data.max(1)[1].cpu().numpy()
                #sigmoid on outputs
                target = F.upsample(outputs, size=(args.img_rows, args.img_cols), mode='bilinear')
                target_nhwc = target.transpose(1, 2).transpose(2, 3)
                # target=target_nhwc.view(-1,1)
                #labels_3=labels.unsqueeze(-1)
                pred=htan(target_nhwc).data.cpu()
                pred_nchw=htan(target)
                labels_nchw=labels_val.transpose(3,2).transpose(2,1)
                fg_loss,bg_loss=jloss_fn(pred_nchw,labels_nchw)
                # loss=(0.4*fg_loss) + (0.6*bg_loss)
                gt = labels_val.cpu()
                #sdissim=1-ssim_loss(target_nhwc,labels_3)
                g_loss=gloss(pred_nchw, labels_nchw)
                loss = loss_fn(pred, gt)
                val_loss+=float(loss)
                val_mse+=float(MSE(pred, gt))
                val_gloss+=float(g_loss)
                val_fg+=float(fg_loss)
                val_bg+=float(bg_loss)
            

            if args.visdom:
                choices=random.sample(range(images_val.shape[0]), 5)
                #show batch output and labels
                out_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)
                inp_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)
                label_arr=np.full((5,3,args.img_rows,args.img_cols),0.0)  
                
                labels_nchw=labels_val.transpose(3,2).transpose(2,1)
                target_cpu=pred_nchw.detach().cpu().numpy()
                labels_cpu=labels_nchw.detach().cpu().numpy()
                inp_cpu=images_val.detach().cpu().numpy()
                idx=0
                for c in choices:
                    # print(labels_nchw.shape)
                    out_arr[idx,:,:,:]=target_cpu[c]
                    label_arr[idx,:,:,:]=labels_cpu[c]
                    inp_arr[idx,0,:,:]=inp_cpu[c][0,:,:]
                    inp_arr[idx,1,:,:]=inp_cpu[c][1,:,:]
                    inp_arr[idx,2,:,:]=inp_cpu[c][2,:,:]
                    
                    # print(np.max(inp_arr[idx]))
                    # print(np.min(inp_arr[idx]))
                    idx+=1
                
                vis.images(out_arr,
                           win=val_out_win,
                           opts=dict(title='Val Outputs', caption='WcvarX_'+experiment_name))
                vis.images(label_arr,
                           win=val_labels_win,
                           opts=dict(title='Val Labels', caption='WcvarX_'+experiment_name))
                vis.images(inp_arr,
                           win=val_inputs_win,
                           opts=dict(title='Val Inputs', caption='WcvarX_'+experiment_name))
        # break

        #     running_metrics.update(gt, pred)

        # score, class_iou = running_metrics.get_scores()
        # for k, v in score.items():
        #     print(k, v)
        # running_metrics.reset()

        print("val loss at epoch {}:: {}".format(epoch+1,val_loss/len(valloader)))
        val_mse=val_mse/len(valloader)
        print("val MSE: {}".format(val_mse))

        val_losses=[val_loss/len(valloader), val_mse, val_fg/len(valloader),val_bg/len(valloader),val_gloss/len(valloader)]
        write_log_file(experiment_name, val_losses, epoch+1, lrate, 'Val')

        #reduce learning rate
        sched.step(val_mse) 
        
        if val_mse < best_val_mse:
            best_val_mse=val_mse
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "./checkpoints-swat3d/{}_{}_{}_{}_{}_{}_best_model.pkl".format(args.arch, args.dataset, epoch+1,val_mse,train_mse,experiment_name))

        if (epoch+1) % 10 == 0:
            # best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "./checkpoints-swat3d/{}_{}_{}_{}_{}_{}_model.pkl".format(args.arch, args.dataset, epoch+1,val_mse,train_mse,experiment_name))

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
