import os
import math
import random
import numpy as np
import pandas as po
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim

from nn_utils import train
from force_field import load_config, generate_force_dataset
from Hnet import Hnet
from Hmat import Hmat
from weight_init_schemes import weight_init_constant, weight_init_uniform, weight_init_normal, weight_init_xavier

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(args):
    # set seed
    print("Using seed = {}".format(args.seed))
    set_seeds(args.seed)

    # set torch device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device = {}".format(args.device))

    # initialize network
    num_model = 5
    nets = []
    for i in range(num_model):
        nets.append(Hnet(z_dim=args.z_dim, model=args.env))

    # weight initialization
    #net.apply(weight_init_xavier)

    # load dataset
    #mode = "numerical"
    mode = "symbolic"
    f_z, z = generate_force_dataset(args, mode=mode)
    data = torch.cat([z.clone(), f_z.clone()], dim=1).detach().numpy()
    print(data.shape)
    np.savetxt("./data_samples/{}.txt".format(args.env), data)

    '''# train net
    if (args.train):
        loss = train(nets, f_z, z, args.SAVE_PATH, epochs=args.epochs, lr=args.learning_rate, bs=args.batch_size, es_delta=args.es_delta, max_loss_divergence=args.max_loss_divergence, weight_decay=args.weight_decay, device=args.device, model=args.env) #, weight_decay_decay=args.weight_decay_decay)'''


if __name__ == '__main__':
    args = load_config()
    main(args)
