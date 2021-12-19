import torch, os, sys, cv2, json, argparse, random
import torch.nn as nn
from torch.nn import init
import functools
import torch.optim as optim
import torchvision.models as tvmodels

from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as func
from PIL import Image

import torchvision.transforms as transforms
import numpy as np 


class VGG19_5_4(nn.Module):

	def __init__(self):
		super(VGG19_5_4, self).__init__()
		features = list(tvmodels.vgg16(pretrained=True).cuda().features)[:36]

		self.features = nn.ModuleList(features).eval()

	def forward(self, x):
		for ii, model in enumerate(self.features):
			x = model(x)

		return x


class VGG16_3_3(nn.Module):

	def __init__(self):
		super(VGG16_3_3, self).__init__()
		features = list(tvmodels.vgg16(pretrained=True).cuda().features)[:16]

		self.features = nn.ModuleList(features).eval()

	def forward(self, x):
		for ii, model in enumerate(self.features):
			x = model(x)

		return x

class PerceptualLoss(nn.Module):

	def __init__(self):
		super().__init__()

		self.vgg = VGG16_3_3()
		self.l2 = nn.MSELoss()
		self.l1 = nn.L1Loss()

	def forward(self, output, target):
		output_vgg = self.vgg(output)
		target_vgg = self.vgg(target)

		return 0.8*self.l1(output, target) + 0.2*self.l2(output_vgg, target_vgg)
