#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.decomposition import PCA
import torch.optim as optim
import copy
import argparse



def initialize():
    session = {}
    session['prepca'] = "no"
    session['noise_threshold'] = "0.01"
    session['hidden_widths'] = "[200,200]"
    session['slope'] = "0.0"
    session['L'] = "[0.01,0.02,0.05,0.075,0.1,0.15,0.2,0.3,0.5,1.0,2.0]"
    session['opt'] = "Adam"
    session['learning_rate'] = "0.001"
    session['batch_size'] = "256"
    session['training_iteration'] = "2000"
    session['a'] = "2"
    session['n_walk'] = "2000" 
    return session


def train(session, model):
    
    #<--------------Pass Parameters-------------->
    prepcaremove = session['prepca']
    noise_threshold = float(session['noise_threshold'])
    nn_widths = list(eval(session['hidden_widths']))
    hidden_depth = len(nn_widths)
    slope = float(session['slope'])
    sigmals = eval(session['L'])
    opt = session['opt']
    lr = float(session['learning_rate'])
    batch_size = int(session['batch_size'])
    epoch = int(session['training_iteration'])
    a = float(session['a'])
    n_walk = int(session['n_walk'])
    log = 200
    
    #<--------------Load data-------------->
    xs = np.loadtxt("./saved_checkpoints/{}/Hmat/Hmat.txt".format(model))
    n_train = xs.shape[0]
    input_dim = xs.shape[1]
    
    #<--------------Preprocessing-------------->
    # normalize each H
    xs = (xs - np.mean(xs, axis=0)[np.newaxis,:])
    xs = xs/np.std(xs, axis=0)[np.newaxis,:]
    
    pca = PCA()
    pca.fit_transform(xs)
    #pca.fit(xs)
    print(pca.singular_values_**2/np.sum(pca.singular_values_**2))
    
    remove_dim = 0
    if prepcaremove == "no":
        #xs = xs/(np.std(xs,axis=0)[np.newaxis,:])
        pass
    else:
        input_dim_orig = input_dim
        input_dim = np.sum(pca.explained_variance_ratio_>noise_threshold)
        remove_dim = input_dim_orig - input_dim
        xs = xs[:,:input_dim]
        xs = xs/(np.std(xs,axis=0)[np.newaxis,:])

    nn_widths.insert(0, input_dim)
    nn_widths.append(input_dim)
        
    #<--------------Build Networks-------------->
    class den(nn.Module):
        def __init__(self):
            super(den, self).__init__()
            self.linears = nn.ModuleList([nn.Linear(nn_widths[i], nn_widths[i+1]) for i in range(hidden_depth+1)])

        def forward(self, x):
            #act = nn.LeakyReLU(slope)
            act = nn.SiLU()
            self.x = x
            for i in range(hidden_depth):
                self.x = act(self.linears[i](self.x))
            self.x = self.linears[hidden_depth](self.x)
            return self.x
    
    #<--------------Training------------->
    exps = []
    losses = []
    os.makedirs('saved_checkpoints/{}/NEB/'.format(model), exist_ok=True)

    for sigmal in sigmals:
        den_net = den()
        criterion = nn.MSELoss()
        if opt == "Adam":
            optimizer = optim.Adam(den_net.parameters(), lr = lr)
        else:
            optimizer = optim.SGD(den_net.parameters(), lr = lr)
        print("sigmal={}".format(sigmal))

        for j in range(epoch):
            den_net.train()
            optimizer.zero_grad()
            choices = np.random.choice(n_train, batch_size)
            perturb = torch.normal(0,sigmal,size=(batch_size,input_dim))
            inputs0 = torch.tensor(xs[choices], dtype=torch.float) + perturb
            outputs = den_net(inputs0)
            loss = criterion(outputs, -perturb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data))

            if j%log == 0:
                print('Epoch:  %d | Train: %.3g' %(j, loss))

        #x0 = copy.deepcopy(xs[int(n_train/2)])
        x0 = copy.deepcopy(xs[2])
        x0 = x0[np.newaxis,:]

        x0 = x0 + np.random.randn(n_walk,input_dim) * sigmal
        x0 = x0 + den_net(torch.tensor(x0,dtype=torch.float)).detach().numpy()

        pca = PCA()
        pca.fit(x0)
        svs = pca.singular_values_
        exp_ratio = svs**2/np.sum(svs**2)
        exps.append(exp_ratio)

        torch.save(den_net.state_dict(), "./saved_checkpoints/{}/NEB/NEB_".format(model)+"%.3f"%sigmal)

    exps = np.array(exps)
    
    #<--------------Plotting------------->
    # ERD
    def f(x,a=2):
        n = x.shape[1]
        mask = x < 1/(a*n)
        return np.sum((1-np.cos(np.pi/2*n*a*x)*mask)**2,axis=1)
    
    ax1 = plt.figure(figsize=(7,5))
    exps = np.array(exps)
    for i in range(input_dim):
        plt.plot(sigmals, exps[:,i], marker="o", color="black", ls="--")
    plt.xscale('log')
    plt.xlabel(r"$L$",fontsize=25)
    plt.ylabel("Explained Ratio",fontsize=25)

    ax2 = ax1.gca().twinx()
    neffs = f(exps, a=a)
    ax2.plot(sigmals, neffs, marker="o",color="red",linewidth=5, markersize=15)
    plt.ylabel(r"$n_{eff}$",fontsize=25,color="red")
	#ax2.set_ylim(0,x.shape[1])
    plt.savefig('./saved_checkpoints/{}/NEB/ERD.png'.format(model),bbox_inches="tight")
    np.savetxt('./saved_checkpoints/{}/NEB/sigmals.txt'.format(model), sigmals)
    np.savetxt('./saved_checkpoints/{}/NEB/exp_ratio.txt'.format(model), exps)
    np.savetxt('./saved_checkpoints/{}/NEB/Neff_L.txt'.format(model), neffs)
    plt.clf()
    # Neff histogram
    den_nets = []
    for j in range(len(sigmals)):
        sigmal = sigmals[j]
        den_net = den()
        den_net.load_state_dict(torch.load("./saved_checkpoints/{}/NEB/NEB_".format(model)+"%.3f"%sigmal))
        den_nets.append(copy.deepcopy(den_net))
        
    exp_ratioss = []
    npoint = 100

    for i in range(npoint):
        if i % 20 == 0:
            print(i)
        iid = np.random.choice(n_train)
        x0 = copy.deepcopy(xs[iid])
        x0 = x0[np.newaxis,:]

        exp_ratios = []

        for j in range(len(sigmals)):

            x0 = x0 + np.random.randn(n_walk,input_dim) * sigmals[j]
            x0 = x0 + den_nets[j](torch.tensor(x0,dtype=torch.float)).detach().numpy()

            pca = PCA()
            pca.fit(x0)
            svs = pca.singular_values_
            exp_ratio = svs**2/np.sum(svs**2)
            exp_ratios.append(exp_ratio)
        exp_ratioss.append(exp_ratios)
    exp_ratioss = np.array(exp_ratioss)
    
    a = np.min(f(exp_ratioss.reshape(-1,input_dim)).reshape(npoint, len(sigmals)),axis=1)
    a = np.round(a).astype('int')
    counts = np.bincount(a)
    neff = np.argmax(counts)
    confidence = float(np.max(counts))/100.
    plt.hist(a,bins=25)
    plt.xlabel(r"$n_{eff}$",fontsize=20)
    plt.ylabel("Count",fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('./saved_checkpoints/{}/NEB/Neff_dist.png'.format(model),bbox_inches="tight")
    np.savetxt('./saved_checkpoints/{}/NEB/Neff_dist.txt'.format(model), a)
    plt.clf()

    return neff, confidence


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help="model")
    args = parser.parse_args()
    
    session = initialize()
    train(session, args.model)


# In[ ]:




