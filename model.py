# -*- coding: UTF-8 -*-
"""
@Project ：Classified 
@File ：model.py
@Author ：AnthonyZ
@Date ：2022/8/25 10:52
"""

import torchvision.models as models
import torch.nn as nn


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class Resnet18Rnn(nn.Module):
	def __init__(self, opt):
		super(Resnet18Rnn, self).__init__()
		self.BaseModel = models.resnet18(pretrained=True)
		num_input = self.BaseModel.fc.in_features
		self.BaseModel.fc = Identity()
		self.lstm = nn.LSTM(num_input, opt.hidden_size, opt.num_layers)
		self.linear = nn.Linear(opt.hidden_size, opt.num_class)
		self.drop_out = nn.Dropout(0.1)
		self.relu = nn.ReLU()
		self.opt = opt

	def forward(self, x):
		batch_size, num_frame, channel, height, width = x.shape
		x = x.view(batch_size * num_frame, channel, height, width)
		x = self.BaseModel(x)
		x = x.view(batch_size, num_frame, -1)
		x, _ = self.lstm(x)
		output = x[:, -1]
		output = self.drop_out(output)
		output = self.linear(output)
		return output.squeeze(1)
