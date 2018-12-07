#implementation of chamfer loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# np.set_printoptions(threshold=np.nan)
from math import exp
import cv2


def get_pairwise_dist(x,y):

	bs, num_points_x, points_dim = x.size()
	_, num_points_y, _ = y.size()
	
	xx=torch.bmm(x, x.transpose(2,1))
	yy=torch.bmm(y, y.transpose(2,1))
	xy=torch.bmm(x, y.transpose(2,1))

	if x.is_cuda:
		dtype = torch.cuda.LongTensor
	else:
		dtype = torch.LongTensor

	diag_ind_x = torch.arange(0, num_points_x).type(dtype)
	diag_ind_y = torch.arange(0, num_points_y).type(dtype) #change here if y has diff points 

	rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
	ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
	P = (rx.transpose(2,1) + ry - 2*zz)

	return P



class Chamferloss(torch.nn.Module):
	def __init__(self):
		super(Chamferloss, self).__init__()

	def forward(self,pred,label):
		batch_size, h, w, channel = pred.size()
		
		#pred is x and label is y

		b_losses=0.0

		#n,h,w,c -> n,h.w,c
		xb=pred.view(-1,h*w,channel)
		yb=label.view(-1,h*w,channel)

		# print (x.shape)

		for b in range (batch_size):
			# discard zeros
			x=xb[b,:,:]
			y=yb[b,:,:]
			
			pairwise_dist= get_pairwise_dist(x.unsqueeze(0),y.unsqueeze(0))

			xmins = torch.min(P, 1)
			loss_1 = torch.sum(xmins)
			ymins = torch.min(P, 2)
			loss_2 = torch.sum(ymins)
			loss=loss_1 + loss_2

			b_losses+=loss

		b_losses=b_losses/batch_size