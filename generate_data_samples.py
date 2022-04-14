import os
import math
import numpy as np
import pandas as po
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def harmonic_1d():
    num_pts = 1000

    data = po.DataFrame()
    data['x'] = 4*np.random.rand(num_pts,)-2
    data['p'] = 4*np.random.rand(num_pts,)-2
    
    data.to_csv('data_samples/harmonic_1d.csv', index=False)
    
   
def harmonic_1d_damp():
    num_pts = 1000

    data = po.DataFrame()
    data['x'] = np.random.normal(loc=0, scale=1, size=num_pts)
    data['p'] = np.random.normal(loc=0, scale=1, size=num_pts)
    
    data.to_csv('data_samples/harmonic_1d_damp.csv', index=False)



def harmonic_2d_iso():
    num_pts = 10000

    data = po.DataFrame()
    #data['x'] = np.random.normal(loc=0, scale=1, size=num_pts)
    #data['p_x'] = np.random.normal(loc=0, scale=1, size=num_pts)
    #data['y'] = np.random.normal(loc=0, scale=1, size=num_pts)
    #data['p_y'] = np.random.normal(loc=0, scale=1, size=num_pts)
    data['x'] = 4*np.random.rand(num_pts,)-2
    data['p_x'] = 2*np.random.rand(num_pts,)-2
    data['y'] = 4*np.random.rand(num_pts,)-2
    data['p_y'] = 4*np.random.rand(num_pts,)-2
    
    data.to_csv('data_samples/harmonic_2d_iso.csv', index=False)


def harmonic_2d_aniso():

    num_pts = 10000
    
    data = po.DataFrame()
    #data['x'] = np.random.normal(loc=0, scale=1, size=num_pts)
    #data['p_x'] = np.random.normal(loc=0, scale=1, size=num_pts)
    #data['y'] = np.random.normal(loc=0, scale=1, size=num_pts)
    #data['p_y'] = np.random.normal(loc=0, scale=1, size=num_pts)
    data['x'] = 4*np.random.rand(num_pts,)-2
    data['p_x'] = 4*np.random.rand(num_pts,)-2
    data['y'] = 4*np.random.rand(num_pts,)-2
    data['p_y'] = 4*np.random.rand(num_pts,)-2
    
    data.to_csv('data_samples/harmonic_2d_aniso.csv', index=False)


def kepler_2d():
    num_pts = 12000

    data = po.DataFrame()
    #data['x'] = np.random.normal(loc=0, scale=1, size=num_pts)
    #data['p_x'] = np.random.normal(loc=0, scale=1, size=num_pts)
    #data['y'] = np.random.normal(loc=0, scale=1, size=num_pts)
    #data['p_y'] = np.random.normal(loc=0, scale=1, size=num_pts)
    data['x'] = 4*np.random.rand(num_pts,)-2
    data['p_x'] = 2*np.random.rand(num_pts,)-1
    data['y'] = 4*np.random.rand(num_pts,)-2
    data['p_y'] = 2*np.random.rand(num_pts,)-1
    radius = np.sqrt(data['x']**2 + data['y']**2)
    select = (radius > 0.5)
    data = data[select]
    
    print("kepler")
    print(np.mean(data['p_x']**2+data['p_y']**2))
    print(np.mean(1/np.sqrt(data['x']**2+data['y']**2)))
    
    data.to_csv('data_samples/kepler_2d.csv', index=False)



def quantum():

    data = po.DataFrame()

    num_pts = 40
    num_pack = 5
    num_wave = 1000
    
    shift = 1*(np.random.rand(num_pack*num_wave,)-0.5)[:,np.newaxis]

    x = np.linspace(-10,10,num=2*num_pts+1)[1:-1:2][np.newaxis,:]

    def p(x, sigma, mu, mags):
        return mags*(1/(np.sqrt(2*np.pi)*sigma))*np.e**(-(x-mu)**2/(2*sigma**2)) + shift

    def px(x, sigma, mu, mags):
        return -(p(x, sigma, mu, mags)-shift)*(x-mu)/sigma**2

    def pxx(x, sigma, mu, mags):
        return (-sigma**2+(x-mu)**2)/sigma**4*(p(x,sigma,mu,mags)-shift)

    def pxxx(x, sigma, mu, mags):
        return (3*(x-mu)*sigma**2+(x-mu)**3)/sigma**6*(p(x,sigma,mu,mags)-shift)

    def pxxxx(x, sigma, mu, mags):
        return (3*sigma**4-6*(x-mu)**2*sigma**2+(x-mu)**4)/sigma**8*(p(x,sigma,mu,mags)-shift)

    def pxxxxx(x, sigma, mu, mags):
        return (-15*(x-mu)*sigma**4+10*(x-mu)**3*sigma**2-(x-mu)**5)/sigma**10*(p(x,sigma,mu,mags)-shift)

    sigmas = np.array([1.5]*num_pack*num_wave)[:,np.newaxis]
    mus = (np.random.rand(num_pack*num_wave,)*6-3)[:,np.newaxis]
    mags = ((np.random.rand(num_pack*num_wave,)-0.5)*10)[:,np.newaxis]
    
    noise = 0.0

    psi_r = np.mean(p(x,sigmas,mus,mags).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)
    psi_r_x = np.mean(px(x,sigmas,mus,mags).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)
    psi_r_xx = np.mean(pxx(x,sigmas,mus,mags).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)
    psi_r_xxx = np.mean(pxxx(x,sigmas,mus,mags).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)
    psi_r_xxxx = np.mean(pxxxx(x,sigmas,mus,mags).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)
    
    sigmas2 = np.array([1.5]*num_pack*num_wave)[:,np.newaxis]
    mus2 = (np.random.rand(num_pack*num_wave,)*6-3)[:,np.newaxis]
    mags2 = ((np.random.rand(num_pack*num_wave,)-0.5)*10)[:,np.newaxis]

    psi_i = np.mean(p(x,sigmas2,mus2,mags2).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)
    psi_i_x = np.mean(px(x,sigmas2,mus2,mags2).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)
    psi_i_xx = np.mean(pxx(x,sigmas2,mus2,mags2).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)
    psi_i_xxx = np.mean(pxxx(x,sigmas2,mus2,mags2).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)
    psi_i_xxxx = np.mean(pxxxx(x,sigmas2,mus2,mags2).reshape(num_pack, num_wave, num_pts), axis=0) + noise*np.random.randn(num_wave,num_pts)

    k = 0
    
    print(np.mean((-1/2*psi_i_xx)**2))
    print(np.mean(((psi_r**2+psi_i**2)*psi_i)**2))
    print(np.mean((1/2*psi_r_xx)**2))
    print(np.mean(((psi_r**2+psi_i**2)*psi_r)**2))
    
    noise2 = 0.0
    

    f_r = -1/2*psi_i_xx + k*(psi_r**2+psi_i**2)*psi_i + noise2*np.random.randn(num_wave,num_pts)
    f_i = 1/2*psi_r_xx - k*(psi_r**2+psi_i**2)*psi_r + noise2*np.random.randn(num_wave,num_pts)
    f_r_x = -1/2*psi_i_xxx + 2*k*(psi_r*psi_r_x+psi_i*psi_i_x)*psi_i + k*(psi_r*2+psi_i**2)*psi_i_x + noise2*np.random.randn(num_wave,num_pts)
    f_i_x = 1/2*psi_r_xxx - 2*k*(psi_r*psi_r_x+psi_i*psi_i_x)*psi_r - k*(psi_r**2+psi_i**2)*psi_r_x + noise2*np.random.randn(num_wave,num_pts)
    f_r_xx = -1/2*psi_i_xxxx + 2*k*(psi_r_x**2+psi_r*psi_r_xx+psi_i_x**2+psi_i*psi_i_xx)*psi_i + 4*k*(psi_r*psi_r_x+psi_i*psi_i_x)*psi_i_x+k*(psi_r**2+psi_i**2)*psi_i_xx + noise2*np.random.randn(num_wave,num_pts)
    f_i_xx = 1/2*psi_r_xxxx - 2*k*(psi_r_x**2+psi_r*psi_r_xx+psi_i_x**2+psi_i*psi_i_xx)*psi_r-4*k*(psi_r*psi_r_x+psi_i*psi_i_x)*psi_r-k*(psi_r**2+psi_i**2)*psi_r_xx + noise2*np.random.randn(num_wave,num_pts)

    #psi = np.concatenate([psi_r[:,:,np.newaxis]**2+psi_i[:,:,np.newaxis]**2], axis=2)
    psi = np.concatenate([psi_r[:,:,np.newaxis]**2+psi_i[:,:,np.newaxis]**2, psi_r_x[:,:,np.newaxis]**2+psi_i_x[:,:,np.newaxis]**2], axis=2)
    #psi = np.concatenate([psi_r[:,:,np.newaxis]**2+psi_i[:,:,np.newaxis]**2, psi_r_x[:,:,np.newaxis]**2+psi_i_x[:,:,np.newaxis]**2,psi_r_xx[:,:,np.newaxis]**2+psi_i_xx[:,:,np.newaxis]**2], axis=2)
    #psi = np.concatenate([psi_r[:,:,np.newaxis],psi_i[:,:,np.newaxis]], axis=2)
    #psi = np.concatenate([psi_r[:,:,np.newaxis],psi_r_x[:,:,np.newaxis],psi_i[:,:,np.newaxis],psi_i_x[:,:,np.newaxis]], axis=2)
    #psi = np.concatenate([psi_r[:,:,np.newaxis],psi_r_x[:,:,np.newaxis],psi_r_xx[:,:,np.newaxis],psi_i[:,:,np.newaxis],psi_i_x[:,:,np.newaxis],psi_i_xx[:,:,np.newaxis]], axis=2)
    

    #f = np.concatenate([f_r[:,:,np.newaxis]*psi_r[:,:,np.newaxis]+f_i[:,:,np.newaxis]*psi_i[:,:,np.newaxis]], axis=2)
    f = np.concatenate([f_r[:,:,np.newaxis]*psi_r[:,:,np.newaxis]+f_i[:,:,np.newaxis]*psi_i[:,:,np.newaxis], f_r_x[:,:,np.newaxis]*psi_r_x[:,:,np.newaxis]+f_i_x[:,:,np.newaxis]*psi_i_x[:,:,np.newaxis]], axis=2)
    #f = np.concatenate([f_r[:,:,np.newaxis]*psi_r[:,:,np.newaxis]+f_i[:,:,np.newaxis]*psi_i[:,:,np.newaxis], f_r_x[:,:,np.newaxis]*psi_r_x[:,:,np.newaxis]+f_i_x[:,:,np.newaxis]*psi_i_x[:,:,np.newaxis], f_r_xx[:,:,np.newaxis]*psi_r_xx[:,:,np.newaxis]+f_i_xx[:,:,np.newaxis]*psi_i_xx[:,:,np.newaxis]], axis=2)
    #f = np.concatenate([f_r[:,:,np.newaxis],f_i[:,:,np.newaxis]], axis=2)
    #f = np.concatenate([f_r[:,:,np.newaxis],f_r_x[:,:,np.newaxis],f_i[:,:,np.newaxis],f_i_x[:,:,np.newaxis]], axis=2)
    #f = np.concatenate([f_r[:,:,np.newaxis],f_r_x[:,:,np.newaxis],f_r_xx[:,:,np.newaxis],f_i[:,:,np.newaxis],f_i_x[:,:,np.newaxis],f_i_xx[:,:,np.newaxis]], axis=2)
    #f = np.concatenate([np.sqrt(f_r[:,:,np.newaxis]**2+f_i[:,:,np.newaxis]**2),np.sqrt(f_r_x[:,:,np.newaxis]**2+f_i_x[:,:,np.newaxis]**2)], axis=2)
    
    data["psi"] = psi.reshape(-1,)
    data["f"] = f.reshape(-1,)

    data.to_csv('data_samples/quantum.csv', index=False)


def KdV_1d():
    data = po.DataFrame()
    num_pts = 40
    num_pack = 5
    num_wave = 10000

    x = np.linspace(-10,10,num=2*num_pts+1)[1:-1:2][np.newaxis,:]

    def p(x, sigma, mu, mags):
        return mags*(1/(np.sqrt(2*np.pi)*sigma))*np.e**(-(x-mu)**2/(2*sigma**2))

    def px(x, sigma, mu, mags):
        return -p(x, sigma, mu, mags)*(x-mu)/sigma**2

    def pxx(x, sigma, mu, mags):
        return (-sigma**2+(x-mu)**2)/sigma**4*p(x,sigma,mu,mags)

    def pxxx(x, sigma, mu, mags):
        return (3*(x-mu)*sigma**2+(x-mu)**3)/sigma**6*p(x,sigma,mu,mags)

    def pxxxx(x, sigma, mu, mags):
        return (3*sigma**4-6*(x-mu)**2*sigma**2+(x-mu)**4)/sigma**8*p(x,sigma,mu,mags)

    def pxxxxx(x, sigma, mu, mags):
        return (-15*(x-mu)*sigma**4+10*(x-mu)**3*sigma**2-(x-mu)**5)/sigma**10*p(x,sigma,mu,mags)
    
    sigmas = np.array([1.5]*num_pack*num_wave)[:,np.newaxis]
    mus = (np.random.rand(num_pack*num_wave,)*6-3)[:,np.newaxis]
    mags = ((np.random.rand(num_pack*num_wave,)-0.5)*15)[:,np.newaxis]

    p_ = p(x,sigmas,mus,mags)
    px_ = px(x,sigmas,mus,mags)
    pxx_ = pxx(x,sigmas,mus,mags)
    pxxx_ = pxxx(x,sigmas,mus,mags)
    pxxxx_ = pxxxx(x,sigmas,mus,mags)
    pxxxxx_ = pxxxx(x,sigmas,mus,mags)

    p_ = np.mean(p_.reshape(num_pack, num_wave, num_pts), axis=0)
    px_ = np.mean(px_.reshape(num_pack, num_wave, num_pts), axis=0)
    pxx_ = np.mean(pxx_.reshape(num_pack, num_wave, num_pts), axis=0)
    pxxx_ = np.mean(pxxx_.reshape(num_pack, num_wave, num_pts), axis=0)
    pxxxx_ = np.mean(pxxxx_.reshape(num_pack, num_wave, num_pts), axis=0)
    pxxxxx_ = np.mean(pxxxxx_.reshape(num_pack, num_wave, num_pts), axis=0)
    
    #print(np.mean((6*p_*px_)**2))
    #print(np.mean((pxxx_)**2))

    f_ = 6*p_*px_ - pxxx_
    fx_ = 6*px_*px_ + 6*p_*pxx_ - pxxxx_
    fxx_ = 12*px_*pxx_ + 6*px_*pxx_ + 6*p_*pxxx_ - pxxxxx_

    #psi = np.concatenate([p_[:,:,np.newaxis],np.abs(px_[:,:,np.newaxis]),np.abs(pxx_[:,:,np.newaxis])], axis=2)
    psi = np.concatenate([p_[:,:,np.newaxis],np.abs(px_[:,:,np.newaxis])], axis=2)
    #psi = np.concatenate([p_[:,:,np.newaxis]], axis=2)

    #f = np.concatenate([f_[:,:,np.newaxis],fx_[:,:,np.newaxis],fxx_[:,:,np.newaxis]], axis=2)
    f = np.concatenate([f_[:,:,np.newaxis],fx_[:,:,np.newaxis]*np.sign(px_[:,:,np.newaxis])], axis=2)
    #f = np.concatenate([f_[:,:,np.newaxis]], axis=2)

    print(psi.shape)
    print(f.shape)
    
    #print(np.mean((6*px_*px_)**2))
    #print(np.mean((6*p_*pxx_)**2))
    #print(np.mean((pxxxx_)**2))

    data["psi"] = psi.reshape(-1,)
    data["f"] = f.reshape(-1,)

    data.to_csv('data_samples/KdV_1d.csv', index=False)

def threebody_2d():
    num_pts = 15000
    dims = 12

    data = po.DataFrame()
    for i in range(1, 12+1):
        data['x_{}'.format(i)] = 4*np.random.rand(num_pts,)-2 #np.random.normal(loc=0, scale=1, size=num_pts)
        
    r12 = np.sqrt((data['x_1']-data['x_5'])**2 + (data['x_2']-data['x_6'])**2)
    r13 = np.sqrt((data['x_1']-data['x_9'])**2 + (data['x_2']-data['x_10'])**2)
    r23 = np.sqrt((data['x_5']-data['x_9'])**2 + (data['x_6']-data['x_10'])**2)
    select = (r12 > 0.5) & (r13 > 0.5) & (r23 > 0.5)
    data = data[select]
    print(data.shape[0])

    data.to_csv('data_samples/threebody_2d.csv', index=False)

os.makedirs('data_samples', exist_ok=True)

harmonic_1d()
harmonic_1d_damp()
harmonic_2d_iso()
harmonic_2d_aniso()
kepler_2d()
quantum()
threebody_2d()
KdV_1d()
