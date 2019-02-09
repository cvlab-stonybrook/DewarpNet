#hourglass module with residual connection

import torch.nn as nn
import torch
import src.models
# import src.models.utils
#from src.models import get_model
#from src.models.utils import *

class Hourglass(nn.Module):
	def __init__(self, input_nc, output_nc, module1_model_path=None):
		super(Hourglass, self).__init__()
		self.module1 = src.models.get_model('unetnc',n_classes=output_nc, in_channels=input_nc)
		self.module1 = torch.nn.DataParallel(self.module1, device_ids=range(torch.cuda.device_count()))
		self.module1.cuda()

		if module1_model_path:
			checkpoint = torch.load(module1_model_path)
			self.module1.load_state_dict(checkpoint['model_state'])

			# freeze module1 parameters
			for param in self.module1.parameters():
				param.requires_grad=False

		self.module2 = src.models.get_model('unetnc',n_classes=output_nc, in_channels=input_nc)
		self.module2 = torch.nn.DataParallel(self.module2, device_ids=range(torch.cuda.device_count()))
		self.module2.cuda()


	def forward(self, inputs):			# all are in nchw
		
		module1_out=self.module1(inputs)
		residual=torch.add(module1_out,inputs)
		module2_out=self.module2(residual)

		return module2_out



