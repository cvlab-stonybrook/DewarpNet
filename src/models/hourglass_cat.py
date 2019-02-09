#hourglass module with concatenation

import torch.nn as nn
import torch
import src.models
#from src.models import get_model
#from src.models.utils import *

class HourglassC(nn.Module):
	def __init__(self, input_nc, output_nc, module1_model_path=None):
		super(HourglassC, self).__init__()
		self.module1 = src.models.get_model('unetnc',n_classes=output_nc, in_channels=input_nc)
		self.module1 = torch.nn.DataParallel(self.module1, device_ids=range(torch.cuda.device_count()))
		self.module1.cuda()

		if module1_model_path:
			checkpoint = torch.load(module1_model_path)
			self.module1.load_state_dict(checkpoint['model_state'])

			# freeze module1 parameters
			for param in self.module1.parameters():
				param.requires_grad=False

		self.module2 = src.models.get_model('unetnc',n_classes=output_nc, in_channels=input_nc+output_nc)
		self.module2 = torch.nn.DataParallel(self.module2, device_ids=range(torch.cuda.device_count()))
		self.module2.cuda()


	def forward(self, inputs):			# all are in nchw
		
		module1_out=self.module1(inputs)
		ccat=torch.cat([module1_out,inputs],dim=1)
		module2_out=self.module2(ccat)

		return module2_out



