# Validate wc regression 

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



def write_log_file(filepath,losses, epoch, phase):
	with open(filepath,'a') as f:
		f.write("\n{} Epoch: {} Loss: {} MSE: {} BG: {} FG: {} GradLoss: {}".format(phase, epoch, losses[0], losses[1], losses[2], losses[3], losses[4]))

def val(args):

	#model_name
	model_file_name = os.path.split(args.model)[1]
	epoch=model_file_name.split('_')[2]
	
	#create output directory
	model_out_dir=os.path.join(args.out_path, model_file_name[:-4])
	if not os.path.exists(model_out_dir):
		os.makedirs(model_out_dir)

	#create a directory named epoch
	out_dir=os.path.join(model_out_dir,epoch)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	#setup loader
	data_loader = get_loader(args.dataset+'wc')
	data_path = get_data_path(args.dataset)
	v_loader = data_loader(data_path, is_transform=True, split='valswat3d', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

	n_classes = v_loader.n_classes
	valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

	#setup model
	model = get_model(args.arch, n_classes, model_path=args.module1_model ,in_channels=3)
	model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
	model.cuda()

	#setup activation
	htan = nn.Hardtanh(0,1.0)

	#load checkpoint                                         
	if os.path.isfile(args.model):
		print("Loading model from checkpoint '{}'".format(args.model))
		checkpoint = torch.load(args.model)
		model.load_state_dict(checkpoint['model_state'])
		# optimizer.load_state_dict(checkpoint['optimizer_state'])
		print("Loaded checkpoint '{}' (epoch {})"                    
			  .format(args.model, checkpoint['epoch']))  
	else:
		print("No checkpoint found at '{}'".format(args.model)) 

	#setup log file
	log_file=open(os.path.join(out_dir,'log.txt'),'w+')
	log_file.write('\n---------------  '+model_file_name+'  ---------------\n')
	log_file.close()

	#loss functions
	MSE = nn.MSELoss()
	loss_fn = nn.L1Loss()

	#validate dataset
	model.eval()
	val_loss=0.0
	val_mse=0.0
	val_bg=0.0
	val_fg=0.0
	val_gloss=0.0
	saved=0
	batch_choice=[0,10,20]
	print batch_choice
	for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
		with torch.no_grad():
			images_val = Variable(images_val.cuda())
			labels_val = Variable(labels_val.cuda())

			outputs = model(images_val)
			target = F.upsample(outputs, size=(args.img_rows, args.img_cols), mode='bilinear')
			target_nhwc = target.transpose(1, 2).transpose(2, 3)
			pred=htan(target_nhwc).data.cpu()
			pred_nchw=htan(target)
			labels_nchw=labels_val.transpose(3,2).transpose(2,1)
			# fg_loss,bg_loss=jloss_fn(pred_nchw,labels_nchw)
			# loss=(0.4*fg_loss) + (0.6*bg_loss)
			gt = labels_val.cpu()
			#sdissim=1-ssim_loss(target_nhwc,labels_3)
			# g_loss=gloss(pred_nchw, labels_nchw)
			loss = loss_fn(pred, gt)
			val_loss+=float(loss)
			val_mse+=float(MSE(pred, gt))
			# val_gloss+=float(g_loss)
			# val_fg+=fg_loss
			# val_bg+=bg_loss

		# print i_val
		if args.save and i_val in batch_choice:
			print i_val
			choices=[0,1,2,3,4]
			#show batch output and labels
			labels_nchw=labels_val.transpose(3,2).transpose(2,1)
			target_cpu=pred_nchw.detach().cpu().numpy()
			labels_cpu=labels_nchw.detach().cpu().numpy()
			inp_cpu=images_val.detach().cpu().numpy()
			idx=0
			for c in choices:
				# print(labels_nchw.shape)
				# out_arr=np.full((3,args.img_rows,args.img_cols),0.0)
				# inp_arr=np.full((3,args.img_rows,args.img_cols),0.0)
				# label_arr=np.full((3,args.img_rows,args.img_cols),0.0) 
				out_arr=target_cpu[c]
				label_arr=labels_cpu[c]
				inp_arr=inp_cpu[c]
				
				# print(np.max(inp_arr[idx]))
				# print(np.min(inp_arr[idx]))
				
				#save the images
				out_arr=(out_arr.transpose(1,2,0))*255
				label_arr=(label_arr.transpose(1,2,0))*255
				inp_arr=(inp_arr.transpose(1,2,0))*255

				out = Image.fromarray(out_arr.astype(np.uint8))
				out.save(os.path.join(out_dir,'out_'+str(i_val)+'_'+str(idx)+'.png'))
				label = Image.fromarray(label_arr.astype(np.uint8))
				label.save(os.path.join(out_dir,'label_'+str(i_val)+'_'+str(idx)+'.png'))
				inp = Image.fromarray(inp_arr.astype(np.uint8))
				inp.save(os.path.join(out_dir,'inp_'+str(i_val)+'_'+str(idx)+'.png'))
				idx+=1

			saved+=1
			print("Saved...{}/{}".format(saved,args.nimages))


	print("val loss at epoch {}:: {}".format(int(epoch)+1,val_loss/len(valloader)))
	val_mse=val_mse/len(valloader)
	print("val MSE: {}".format(val_mse))

	val_losses=[val_loss/len(valloader), val_mse, val_fg/len(valloader),val_bg/len(valloader),val_gloss/len(valloader)]
	write_log_file(os.path.join(out_dir,'log.txt'), val_losses, int(epoch)+1, 'Val')



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

	parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
						help='Batch Size')
	
	parser.add_argument('--nimages', nargs='?', type=int, default=1, 
						help='Number of batches to save')
	
	parser.add_argument('--model', nargs='?', type=str, default=None,    
						help='Path to previous saved model to validate')
	parser.add_argument('--module1_model', nargs='?', type=str, default=None,    
						help='Path to previous saved model of module1')

	parser.add_argument('--out_path', nargs='?', type=str, default='./results',    
						help='Path to save images')

	parser.add_argument('--save', dest='save', action='store_true', 
						help='save outputs | True by default')
	parser.set_defaults(save=True)



	args = parser.parse_args()
	val(args)



	#CUDA_VISIBLE_DEVICES=1 python valWc.py --model ./checkpoints-swat3d/hourglass_cat_swat3d_1_0.00110028663879_0.00199729626412_hgc_htan_swat3dWcRGBvarYvarXvarXY_l1_bghsaug_retrainFixedvYvXl1grad_best_model.pkl --nimages 5 --arch hourglass_cat --dataset swat3d --img_rows 256 --img_cols 256 --img_norm --module1_model ./checkpoints-swat3d/unetnc_swat3d_319_0.00101629343065_0.000858071765896_htan_swat3dWcRGBvarYvarX_l1grad0.7_bghsaug_retrainFixedvYvXl1_best_model.pkl --batch_size 39
