import os
import math
import glob
import random
import numpy as np
import pandas as po
from tqdm import tqdm
import scipy.stats as ss
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim

from force_field import generate_force_dataset, load_config
from Hnet import Hnet


def Hmat(model, z_dim=4):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model_files = glob.glob('saved_checkpoints/{}/seed_experiments/*.pt'.format(model))
	SAVE_PATH = 'saved_checkpoints/{}/Hmat/'.format(model)
	
	os.makedirs(SAVE_PATH, exist_ok=True)
	f_z, z = generate_force_dataset(args) 
	data_path = 'data_samples/{}.csv'.format(model)
	#z = torch.tensor(po.read_csv(data_path).values, dtype=torch.float, requires_grad=True).to(device)
	print(z.shape)

	df = po.DataFrame()
	for j, mfile in enumerate(model_files):
		print(j)
		net = Hnet(z_dim=z_dim, model=model)
		net.load_state_dict(torch.load(mfile))
		net.to(device)
		h_z = net.h_z(z)
		df[j] = h_z[:,0].detach().numpy()
        
	np.savetxt(SAVE_PATH+'Hmat.txt', df.values)
	
	df.to_csv(SAVE_PATH+'Hmat.csv', index=False)
	
if __name__ == "__main__":
	args = load_config()
	Hmat(model=args.env, z_dim=args.z_dim)
