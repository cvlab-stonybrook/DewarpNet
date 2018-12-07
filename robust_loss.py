# implementation of berhu loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# np.set_printoptions(threshold=np.nan)
from math import exp
import cv2

class BerHuloss(torch.nn.Module):
	def __init__(self):
		super(BerHuloss, self).__init__()

	def forward(self,pred,label):
		batch_size, channel, h, w = pred.size()
		
		# calculate c
		abs_error=torch.sub(pred,label).abs()
		max_abs_error=torch.max(abs_error)
		c=0.2*max_abs_error
		# print (c)

		# 2nd loss component
		other_comp=(torch.pow(abs_error,2)+c**2 )/ (c*2)
		# print torch.max(other_comp)
		# print torch.min(other_comp)
		
		elem_loss=torch.where(abs_error<=c,abs_error,other_comp)

		return torch.mean(elem_loss)


class HuBerloss(torch.nn.Module):
	def __init__(self):
		super(HuBerloss, self).__init__()

	def forward(self,pred,label):
		batch_size, channel, h, w = pred.size()
		
		# calculate c
		abs_error=torch.sub(pred,label).abs()
		max_abs_error=torch.max(abs_error)
		c=0.2*max_abs_error
		# print (c)

		# 2nd loss component
		other_comp=(torch.pow(abs_error,2)+c**2 )/ (c*2)
		# print torch.max(other_comp)
		# print torch.min(other_comp)
		
		elem_loss=torch.where(abs_error<=c,other_comp,abs_error)

		return torch.mean(elem_loss)
