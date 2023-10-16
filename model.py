import torch
import numpy as np
from utils import *

class ParamEstimator:
    def __init__(self,Bas,Vars,bases):
        ### Initialize the estimator with proper values
        # ------------------------------------------------- #
        # Bas: true mu values
        # Vars: true sigma values
        # bases: bases for the function of alpha

        stds = np.sqrt(Vars)
        self.Bas = torch.tensor(Bas)                        # true mu
        self.Sig = torch.tensor(stds)                       # true sigma
        self.bas = torch.tensor(Bas,requires_grad=True)     # estimated mu
        self.sig = torch.tensor(stds,requires_grad=True)    # estimated sigma
        self.bases = bases
        self.T = len(self.Bas)

        # Initialize each parameters in the model
        self.A = torch.tensor([.5,.8],requires_grad=True) # A*Na*Dt
        self.B_T = torch.tensor([0.1,],requires_grad=True) # B_T
        self.B_H = torch.tensor([.5],requires_grad=True)   # B_H
        self.alphas = torch.zeros(len(bases))
        self.alphas[:5] = torch.tensor([.5,0.3,0.2,0.,0.,])  # Initialize alpha
        self.alphas.requires_grad = True

        # Four coefficients corresponding to four loss terms
        self.cd1 = calculate_var(self.Bas)
        self.cd2 = calculate_var(self.Sig)
        self.ce1 = calculate_var(abs(self.Bas[1:]-self.Bas[:-1]))
        self.ce2 = calculate_var(abs(self.Sig[1:]-self.Sig[:-1]))

    def return_A(self,t):
        # return A for given time points t
        # A = A0 * exp(c*t/T)
        return self.A[0] * torch.exp((t/self.T) * self.A[1])

    def return_alpha(self,t):
        # return alpha for given time points t
        alpha = torch.ones_like(t)*self.alphas[0]
        for i in range(1,len(self.alphas)):
            f = self.bases[i]
            alpha += self.alphas[i]*f(2*t/self.T-1)
        return torch.relu(alpha)
    
    def L_d1(self):
        return (self.bas-self.Bas).square().mean()/self.cd1
    
    def L_d2(self):
        return (self.Sig-self.sig).square().mean()/self.cd2
    
    def L_e1(self):
        ts = torch.arange(0,self.T-1)
        alpha = self.return_alpha(ts)
        bt = self.B_T[0]
        bh = self.B_H[0]
        dba = (self.bas[1:]-self.bas[:-1])
        a = self.return_A(ts)
        bas = self.bas[:-1]
        dba1 = -2 * bas * a * (1 - 2*bas/ bt) * torch.exp(-(2*bas/bh).square()) - alpha * bas
        return (dba-dba1).square().mean()/self.ce1
    
    def L_e2(self):
        spi = np.sqrt(np.pi)
        ts = torch.arange(0,self.T-1)
        alpha = self.return_alpha(ts)
        a = self.return_A(ts)
        bt = self.B_T[0]
        sig = self.sig[:-1]
        dvdt = (self.sig[1:]-self.sig[:-1])
        dvdt1 = sig * (-a + 4*a*sig/spi/bt - alpha) 
        return (dvdt-dvdt1).square().mean()/self.ce2

    def train_epoch(self,optimizer,lam1,lam2,lam3,lam4):
        optimizer.zero_grad()
        ld1 = self.L_d1()
        ld2 = self.L_d2()
        le1 = self.L_e1()
        le2 = self.L_e2()
        loss = lam1*ld1 + lam2*ld2 + lam3*le1 + lam4*le2 
        loss.backward()
        optimizer.step()
        return ld1.item(), ld2.item(), le1.item(), le2.item()
    
    def train(self,epochs,lr,lam1,lam2,lam3,lam4):
        optimizer = torch.optim.Adam([self.bas,self.sig,self.A,self.B_T,self.B_H,self.alphas],lr=lr)
        res = []

        for epoch in range(1,epochs+1):
            l1,l2,l3,l4 = self.train_epoch(optimizer,lam1,lam2,lam3,lam4)
            if epoch%100==0:
                a =self.A.detach().numpy()
                bh = self.B_H.detach().numpy()
                bt = self.B_T.detach().numpy()
                alpha = self.alphas.detach().numpy()
                print(f'Epoch {epoch}, Loss,{l1:4e},{l2:4e},{l3:4e},{l4:4e}, a {a}, bh {bh}, bt {bt}, alpha {alpha}')
                res.append([epoch,l1,l2,l3,l4])
        
        return np.array(res)