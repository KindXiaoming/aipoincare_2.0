# weight init schemes

import torch
from torch import nn

def weight_init_constant(m):
	if isinstance(m, nn.Linear):
		nn.init.constant_(m.weight.data, 1.0)
		m.bias.data.zero_()

def weight_init_uniform(m):
	if isinstance(m, nn.Linear):
		n = m.in_features
		y = 1.0/np.sqrt(n)
		m.weight.data.uniform_(-y, y)
		m.bias.data.zero_()

def weight_init_normal(m):
	if isinstance(m, nn.Linear):
		n = m.in_features
		m.weight.data.normal_(0.0, 1.0/np.sqrt(n))
		m.bias.data.zero_()

def weight_init_xavier(m):
	if isinstance(m, nn.Linear):
		nn.init.xavier_uniform_(m.weight.data)
		# Some layers may not have biases
		if (m.bias!=None):
			m.bias.data.zero_()
