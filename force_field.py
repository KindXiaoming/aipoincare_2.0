import os
import numpy as np
import pandas as po
import configargparse
import torch

def load_config():
    p = configargparse.ArgParser()

    # Environment
    p.add('-config', required=True, is_config_file=True, help='config file path')
    p.add('--env', type=str, required=False)
    p.add('--z_dim', type=int, required=True)
    p.add('--data', type=str, required=False)
    p.add('--sparse_H', default=False, action='store_true')

    # Seeds
    p.add('--seed', type=int, default=42, required=False)

    # Actions
    p.add('--train', required=False, default=True, action='store_true')

    # Training parameters
    p.add('--epochs', type=int, required=True)
    p.add('--learning_rate', type=float, required=True)
    p.add('--batch_size', type=int, required=True)
    p.add('--max_loss_divergence', type=float, required=True)
    p.add('--es_delta', type=float, required=True)
    p.add('--weight_decay', type=float, required=False, default=0) 
    #p.add('--weight_decay_decay', required=False, default=False, action='store_true') 

    # System Parameters
    p.add('--k', type=int, required=False)
    p.add('--a', type=int, required=False)
    p.add('--V', type=float, required=False) # V in infinite_well_1D
    p.add('--damping_factor', type=float, required=False) # damping_factor in 1D Harmonic
    p.add('--G', type=float, required=False) # G in three body problem
    p.add('--m1', type=float, required=False) # m1 in three body problem
    p.add('--m2', type=float, required=False) # m2 in three body problem
    p.add('--m3', type=float, required=False) # m3 in three body problem

    args = p.parse_args()

    # Documentation (filepaths in caps)
    os.makedirs('saved_checkpoints', exist_ok=True)
    args.SAVE_DIR = 'saved_checkpoints/{}'.format(args.env)
    os.makedirs(args.SAVE_DIR, exist_ok=True)
    os.makedirs(args.SAVE_DIR+'/seed_experiments/', exist_ok=True)
    args.SAVE_PATH = args.SAVE_DIR+'/seed_experiments'

    return args

def generate_force_dataset(args, mode=""):
    df = po.read_csv(args.data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (args.env=='harmonic_1d'):
        df['0'] = df['p']
        df['1'] = -1*df['x']

        df = df.sample(frac=1, random_state=42)

        f_z = torch.tensor(df[['0', '1']].values, requires_grad=True).float().to(device)
        z = torch.tensor(df[['x', 'p']].values, requires_grad=True).float().to(device)

    elif (args.env=='harmonic_1d_damp'):
        df['dx/dt'] = df['p']
        df['dp/dt'] = -df['x'] - args.damping_factor*df['p']

        df = df.sample(frac=1, random_state=42)

        f_z = torch.tensor(df[['dx/dt', 'dp/dt']].values, requires_grad=True).float().to(device)
        z = torch.tensor(df[['x', 'p']].values, requires_grad=True).float().to(device)

    elif (args.env=='harmonic_2d_iso'):
        df['0'] = df['p_x']
        df['1'] = -1*df['x']
        df['2'] = df['p_y']
        df['3'] = -1*df['y']

        df = df.sample(frac=1, random_state=42)

        f_z = torch.tensor(df[['0', '1', '2', '3']].values, requires_grad=True).float().to(device)
        z = torch.tensor(df[['x', 'p_x', 'y', 'p_y']].values, requires_grad=True).float().to(device)

    elif (args.env=='harmonic_2d_aniso'):
        df['0'] = df['p_x']
        df['1'] = -1*df['x']
        df['2'] = df['p_y']
        df['3'] = -args.k*df['y']

        df = df.sample(frac=1, random_state=42)

        f_z = torch.tensor(df[['0', '1', '2', '3']].values, requires_grad=True).float().to(device)
        z = torch.tensor(df[['x', 'p_x', 'y', 'p_y']].values, requires_grad=True).float().to(device)

            
    elif (args.env=='kepler_2d'):
        df['0'] = df['p_x']
        df['1'] = -df['x']/pow((pow(df['x'], 2)+pow(df['y'], 2)), 1.5)
        df['2'] = df['p_y']
        df['3'] = -df['y']/pow((pow(df['x'], 2)+pow(df['y'], 2)), 1.5)

        df = df.sample(frac=1, random_state=42)

        f_z = torch.tensor(df[['0', '1', '2', '3']].values, requires_grad=True).float().to(device)
        z = torch.tensor(df[['x', 'p_x', 'y', 'p_y']].values, requires_grad=True).float().to(device)
        
        if mode == "symbolic":
            r = torch.sqrt(z[:,[0]]**2 + z[:,[2]]**2)
            f_r = z[:,[0]]/r*f_z[:,[0]] + z[:,[2]]/r*f_z[:,[2]]
            f_z = torch.cat([f_z, f_r], dim=1)
            z = torch.cat([z, r], dim=1)

    elif (args.env=='quantum'):

        f_z = torch.tensor(df['f'].values.reshape(1000,40,args.z_dim), requires_grad=True).float().to(device)
        z = torch.tensor(df['psi'].values.reshape(1000,40,args.z_dim), requires_grad=True).float().to(device)
        
        if mode == "symbolic":
            f_z = f_z.reshape(1000,40*args.z_dim)
            z = z.reshape(1000,40*args.z_dim)

    elif (args.env=='KdV_1d'):

        f_z = torch.tensor(df['f'].values.reshape(10000,40,args.z_dim), requires_grad=True).float().to(device)
        z = torch.tensor(df['psi'].values.reshape(10000,40,args.z_dim), requires_grad=True).float().to(device)
        
        if mode == "symbolic":
            f_z = f_z.reshape(10000,40*args.z_dim)
            z = z.reshape(10000,40*args.z_dim)

    elif (args.env=='threebody_2d'):

        # [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]
        # [x1,y1,v1x,v1y,x2,y2,v2x,v2y,x3,y3,v3x,v3y]

        df['dx_1/dt'] = df['x_3']
        df['dx_5/dt'] = df['x_7']
        df['dx_9/dt'] = df['x_11']

        df['dx_2/dt'] = df['x_4']
        df['dx_6/dt'] = df['x_8']
        df['dx_10/dt'] = df['x_12']

        df['dx_3/dt'] = -args.G*args.m2*(df['x_1']-df['x_5'])/pow(pow(df['x_1']-df['x_5'],2)+pow(df['x_2']-df['x_6'],2),1.5) - args.G*args.m3*(df['x_1']-df['x_9'])/pow(pow(df['x_1']-df['x_9'],2)+pow(df['x_2']-df['x_10'],2),1.5)
        df['dx_7/dt'] = -args.G*args.m1*(df['x_5']-df['x_1'])/pow(pow(df['x_5']-df['x_1'],2)+pow(df['x_6']-df['x_2'],2),1.5) - args.G*args.m3*(df['x_5']-df['x_9'])/pow(pow(df['x_5']-df['x_9'],2)+pow(df['x_6']-df['x_10'],2),1.5)
        df['dx_11/dt'] =  -args.G*args.m1*(df['x_9']-df['x_1'])/pow(pow(df['x_9']-df['x_1'],2)+pow(df['x_10']-df['x_2'],2),1.5) - args.G*args.m3*(df['x_9']-df['x_5'])/pow(pow(df['x_9']-df['x_5'],2)+pow(df['x_6']-df['x_10'],2),1.5)

        df['dx_4/dt'] = -args.G*args.m2*(df['x_2']-df['x_6'])/pow(pow(df['x_1']-df['x_5'],2)+pow(df['x_2']-df['x_6'],2),1.5) - args.G*args.m3*(df['x_2']-df['x_10'])/pow(pow(df['x_1']-df['x_9'],2)+pow(df['x_2']-df['x_10'],2),1.5)
        df['dx_8/dt'] = -args.G*args.m1*(df['x_6']-df['x_2'])/pow(pow(df['x_5']-df['x_1'],2)+pow(df['x_6']-df['x_2'],2),1.5) - args.G*args.m3*(df['x_6']-df['x_10'])/pow(pow(df['x_5']-df['x_9'],2)+pow(df['x_6']-df['x_10'],2),1.5)
        df['dx_12/dt'] =  -args.G*args.m1*(df['x_10']-df['x_2'])/pow(pow(df['x_9']-df['x_1'],2)+pow(df['x_10']-df['x_2'],2),1.5) - args.G*args.m3*(df['x_10']-df['x_6'])/pow(pow(df['x_9']-df['x_5'],2)+pow(df['x_10']-df['x_6'],2),1.5)


        df = df.sample(frac=1, random_state=42)

        z = torch.tensor(df[['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12']].values, requires_grad=True).float().to(device)
        f_z = torch.tensor(df[['dx_1/dt', 'dx_2/dt', 'dx_3/dt', 'dx_4/dt', 'dx_5/dt', 'dx_6/dt', 'dx_7/dt', 'dx_8/dt', 'dx_9/dt', 'dx_10/dt', 'dx_11/dt', 'dx_12/dt']].values, requires_grad=True).float().to(device)
        
        if mode == "symbolic":
            r12 = torch.sqrt((z[:,[0]]-z[:,[4]])**2 + (z[:,[1]]-z[:,[5]])**2)
            r13 = torch.sqrt((z[:,[0]]-z[:,[8]])**2 + (z[:,[1]]-z[:,[9]])**2)
            r23 = torch.sqrt((z[:,[4]]-z[:,[8]])**2 + (z[:,[5]]-z[:,[9]])**2)
            f_r12 = ((z[:,[0]]-z[:,[4]])*(f_z[:,[0]]-f_z[:,[4]])+(z[:,[1]]-z[:,[5]])*(f_z[:,[1]]-f_z[:,[5]]))/r12
            f_r13 = ((z[:,[0]]-z[:,[8]])*(f_z[:,[0]]-f_z[:,[8]])+(z[:,[1]]-z[:,[9]])*(f_z[:,[1]]-f_z[:,[9]]))/r13
            f_r23 = ((z[:,[4]]-z[:,[8]])*(f_z[:,[4]]-f_z[:,[8]])+(z[:,[5]]-z[:,[9]])*(f_z[:,[5]]-f_z[:,[9]]))/r23
            z = torch.cat([z[:,[0,1,2,3]],r12,z[:,[4,5,6,7]],r23,z[:,[8,9,10,11]],r13], dim=1)
            f_z = torch.cat([f_z[:,[0,1,2,3]],f_r12,f_z[:,[4,5,6,7]],f_r23,f_z[:,[8,9,10,11]],f_r13], dim=1)

    else:
        print("Env {} not found!".format(args.env))
        raise AssertionError

    return f_z, z 
