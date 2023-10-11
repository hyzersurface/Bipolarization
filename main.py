from utils import *
from model import ParamEstimator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

if __name__=='__main__':

    data = pd.read_csv('data/HSall_members.csv')

    # preprocess the data
    data['year'] = data['congress']*2 + 1787
    data_h = data[(data['chamber'].isin(['House',]))&(data['congress']>=40)]
    congresses = np.arange(40,119)
    data_h_d = data_h[data_h['party_code']==100]               # Democratic data
    data_h_r = data_h[data_h['party_code']==200]               # Republican data
    data_h_other = data_h[~data_h['party_code'].isin([100,200])]

    feat = 'nominate_dim1'
    mv_d = data_h_d[feat].groupby(data_h_d['congress']).agg(['mean','var'])
    mv_r = data_h_r[feat].groupby(data_h_r['congress']).agg(['mean','var'])
    mv_all = data_h[feat].groupby(data_h['congress']).agg(['mean','var'])
    Na = data_h['congress'].groupby(data_h['congress']).count().mean()/2.
    bas = (mv_r['mean'].values-mv_d['mean'].values)/2          # party polarization
    vars = (mv_d['var'].values+mv_r['var'].values)/2           # party inclusiveness
    stds = np.sqrt(vars)

    # --------------------------------------------------------------------------- #
    # define the estimator with our model
    n = 4 # number of terms of Fourier expansion
    bases = [return_cos(0),]  # Fourier bases
    for i in range(1,n+1):
        bases.append(return_cos(i))
        bases.append(return_sin(i))

    estimator = ParamEstimator(bas,vars,bases) # estimator for the parameters
    epochs = 50000  # number of iterations
    lr = 2e-3       # learning rate
    lam1 = 2.       # lambda_data^mu
    lam2 = 1.       # lambda_data^sigma
    lam3 = 5.       # lambda_eq^mu
    lam4 = 2        # lambda_eq^sigma

    # optimize the loss to learn the parameters
    res = estimator.train(epochs,lr,lam1,lam2,lam3,lam4)


    # ------------------------------------------------------------------#
    ### visualize the results
    ts = np.arange(len(congresses))
    years = (2023 - ts*2)[::-1]
    year_ticks = np.arange(1870,2023,10)
    # learned parameters
    bas_pred = estimator.bas.detach().numpy()
    stds_pred = estimator.vars.detach().numpy()
    a = estimator.A.detach().numpy()
    bh = estimator.B_H.detach().numpy()
    bt = estimator.B_T.detach().numpy()
    alphas = estimator.alphas.detach().numpy()
    print(f'A {a}, Bh {bh}, Bt {bt}, alpha {alphas}')
    with torch.no_grad():
        alpha_learned = estimator.return_alpha(torch.tensor(ts).long())
        a_learned = estimator.return_A(torch.tensor(ts).long())
    
    # plot alpha
    plt.figure(figsize=(2,4))
    plt.plot(alpha_learned,years)
    plt.yticks(year_ticks)
    plt.grid()
    plt.title(r'$\alpha$')
    plt.savefig('figures/HS_two/alpha.jpg',bbox_inches='tight')

    # plot A
    plt.figure(figsize=(2,4))
    plt.plot(a_learned,years)
    plt.yticks(year_ticks)
    plt.grid()
    plt.title(r'$A$')
    plt.savefig('figures/HS_two/a.jpg',bbox_inches='tight')

    # plot polarization
    plt.figure(figsize=(2,4))
    plt.plot(bas,years,label='True')
    plt.plot(bas_pred,years,linestyle='dashed',color='r',label='Learned')
    plt.xlim([0.2,0.5])
    plt.yticks(year_ticks)
    plt.grid()
    plt.legend()
    plt.title('$\mu$')
    plt.savefig('figures/HS_two/mean_learned.jpg',bbox_inches='tight')

    # plot inclusiveness
    plt.figure(figsize=(2,4))
    plt.plot(stds,years,label='True')
    plt.plot(stds_pred,years,linestyle='dashed',color='r',label='Learned')
    plt.xlim([0.05,0.2])
    plt.yticks(year_ticks)
    plt.grid()
    plt.legend()
    plt.title('$\sigma$')
    plt.savefig('figures/HS_two/std_learned.jpg',bbox_inches='tight')
