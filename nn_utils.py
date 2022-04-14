import numpy as np
import pandas as po
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim

def train(nets, f_z, z, SAVE_PATH,
            lr=5e-6,
            epochs = 1000, 
            bs = 100,
            es_delta = 0.000001, 
            epoch_loss = 1e9,
            max_loss_divergence = 0.1,
            device='cuda',
            weight_decay=0, 
            weight_decay_decay=False,model=None):

    num_model = len(nets)
    
    z_dim = z.shape[1]
    entropy_dim = np.minimum(num_model, z_dim)
    parameters = list([])
    for i in range(num_model):
        parameters = list(parameters) + list(nets[i].parameters())
    #optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay, eps=1e-8)
    optimizer = optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    
    for k in range(num_model):
        nets[k].to(device)
        nets[k].train()
    losses = []
    losses_fit = []
    losses_reg = []
    lamb_H = 0.02
    lamb_dH = 0.02
    loss_pow = 2
    

    for epoch in tqdm(range(epochs)):
        for i in range((len(f_z) - 1) // bs + 1):
            choices = np.random.choice(len(f_z), bs, replace=False)
            #start_i = i * bs
            #end_i = start_i + bs
            xb = z[choices, :].to(device)
            fitloss = 0.
            fitlosses = []
            yb = f_z[choices, :].to(device)
            yb = yb.reshape(bs,-1)
            yb_norm = yb/torch.linalg.norm(yb, dim=1, keepdim=True)
            #print("yb", yb_norm.shape)
            for k in range(num_model):
                '''if model == "quantum":
                    pred, pred_t = nets[k](xb)
                    #print("pred",pred.shape)
                    pred_norm = pred/torch.linalg.norm(pred, dim=1, keepdim=True)
                    pred_t_norm = pred_t/torch.linalg.norm(pred_t, dim=1, keepdim=True)
                    fit_orth = torch.matmul(yb_norm.unsqueeze(dim=1), pred_t_norm.unsqueeze(dim=2))
                else:'''
                pred = nets[k](xb)
                #print("pred", pred.shape)
                pred_norm = pred/torch.linalg.norm(pred, dim=1, keepdim=True)
                fit_orth = torch.matmul(yb_norm.unsqueeze(dim=1), pred_norm.unsqueeze(dim=2))
                #print(fit_orth.shape)
                fitloss_ =  torch.mean(torch.abs(fit_orth)**loss_pow)
                #fitloss_each.append(fitloss_.detach().numpy())
                fitloss = fitloss + fitloss_
                fitlosses.append(fitloss_.item())
                H_ = nets[k].h_z(xb)
                dhi = pred_norm
                '''if model == "quantum":
                    dhi_t = pred_t_norm'''
                
                if k == 0:
                    H = H_
                    dH = dhi.unsqueeze(dim=0)
                    '''if model == "quantum":
                        dH_t = dhi_t.unsqueeze(dim=0)'''

                else:
                    H = torch.cat((H,H_), dim=1)
                    dH = torch.cat([dH,dhi.unsqueeze(dim=0)])
                    '''if model == "quantum":
                        dH_t = torch.cat([dH_t,dhi_t.unsqueeze(dim=0)])'''
                #print(dH.shape)
            # fitting loss
            #fitloss = torch.sqrt(fitloss/num_model)
            fitloss = fitloss/num_model
            # correlation loss
            H = torch.transpose(H,0,1)
            # dH.shape = (bs, num_model, z_dim)
            dH = torch.transpose(dH,0,1)
            '''if model == "quantum":
            	dH_t = torch.transpose(dH_t,0,1)'''
            # SVD dH
            u, s, v = torch.svd(dH)
            exp_ratio = s**2/torch.sum(s**2, dim=1,keepdim=True)
            exp_av = torch.mean(exp_ratio,dim=0)
            
            # H regularization
            corr_mat = torch.corrcoef(H)
            if num_model == 1:
                reg_H = torch.tensor(0., dtype=torch.float)
            else:
                reg_H = (torch.sum(corr_mat**loss_pow)-num_model)/(num_model*(num_model-1))
            
            # dH regularization
            #print(dH.shape)
            '''if model == "quantum":
                #print(dH.shape, dH_t.shape)
                orth_mat = torch.mean(torch.abs(torch.matmul(dH_t,torch.transpose(dH,1,2)))**loss_pow, dim=0)**(1/loss_pow)
            else:'''
            orth_mat = torch.mean(torch.abs(torch.matmul(dH, torch.transpose(dH,1,2)))**loss_pow, dim=0)
            #print(orth_mat)
            if num_model == 1:
            	reg_dH = torch.tensor(0., dtype=torch.float)
            else:
            	reg_dH = (torch.sum(orth_mat)-num_model)/(num_model*(num_model-1))

            loss = fitloss + lamb_H*reg_H + lamb_dH*reg_dH
            
            loss.backward()
            optimizer.step()   

            optimizer.zero_grad()

            losses.append(loss.item())
            losses_fit.append(fitloss.item())
            losses_reg.append(reg_dH.item())
            #print(loss.item(), fitloss.item(), reg_dH.item())
         

        # Best epoch loss (save best only)
        if (loss<epoch_loss):
            epoch_loss = loss
            epoch_best = epoch
            fit_loss_best = fitloss
            #print("Saving model at epoch {}, current best epoch loss = {}".format(epoch, epoch_loss))
            if (SAVE_PATH!=None):
                for k in range(num_model):
                    torch.save(nets[k].state_dict(), SAVE_PATH+"/{}.pt".format(k))
                np.save(SAVE_PATH+"/fit_loss.npy", fitlosses)
                np.save(SAVE_PATH+"/dH_explained.npy", exp_av.detach().numpy())

        # Early Stopping
        #if ((loss<es_delta) or abs(epoch_loss-loss)>max_loss_divergence):
        #    break

        # Print loss
        if ((epoch+1)%1==0):
            print("Epoch {}".format(epoch+1))
            print(fitloss.item())
            print(fitlosses)
            print(corr_mat)
            print(reg_H.item())
            print(orth_mat)
            print(reg_dH.item())
            print(exp_av)

    plt.plot(losses)
    plt.plot(losses_fit)
    plt.plot(losses_reg)
    plt.yscale('log')
    if (SAVE_PATH!=None):
        plt.savefig(SAVE_PATH.split('.')[0]+'.png')
        plt.close()
       
    print("Best epoch = {}".format(epoch_best))
    print("Best loss = {}".format(epoch_loss))
    print("Fit loss = {}".format(fit_loss_best))
    df = po.DataFrame()
    df['losses'] = losses
    df.to_csv(SAVE_PATH.split('.')[0]+'_losses.csv', index=False)

    return losses
