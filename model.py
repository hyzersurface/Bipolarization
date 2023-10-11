import torch
import numpy as np
from utils import *

class ParamEstimator:
    def __init__(self,Bas,Vars,bases):
        stds = np.sqrt(Vars)
        self.Bas = torch.tensor(Bas)
        self.Sig = torch.tensor(stds)
        self.bas = torch.tensor(Bas,requires_grad=True)
        self.sig = torch.tensor(stds,requires_grad=True)
        self.bases = bases
        degree = len(bases)
        self.T = len(self.Bas)
        self.A = torch.tensor([.5,.8],requires_grad=True) # ANa
        # cross party, within party
        self.B_T = torch.tensor([0.1,],requires_grad=True)
        self.B_H = torch.tensor([.5],requires_grad=True)
        self.alphas = torch.zeros(degree)
        self.alphas[:5] = torch.tensor([.5,0.3,0.2,0.,0.,])
        self.alphas.requires_grad=True

        self.cd1 = calculate_var(self.Bas)
        self.cd2 = calculate_var(self.Sig)
        self.ce1 = calculate_var(abs(self.Bas[1:]-self.Bas[:-1]))
        self.ce2 = calculate_var(abs(self.Sig[1:]-self.Sig[:-1]))

    def return_A(self,t):
        # A = A0 * exp(c*t/T)
        return self.A[0] * torch.exp((t/self.T) * self.A[1])

    def return_alpha(self,t):
        # alpha = a0/2 + a1*cos(pi*x) + b1*sin(pi*x) + ... + an*cos(n*pi*x) + bn*sin(n*pi*x) + ...
        alpha = torch.ones_like(t)*self.alphas[0]
        for i in range(1,len(self.alphas)):
            f = self.bases[i]
            alpha += self.alphas[i]*f(2*t/self.T-1)
        return torch.relu(alpha)
    
    def L_d1(self):
        return (self.bas-self.Bas).square().mean()/self.cd1
    def L_d2(self):
        return ((self.Sig-self.sig)).square().mean()/self.cd2
    
    def L_e1(self):
        ## dba/dt = -2 A * Na * B_a * (1+2B_a/B_T) * exp(-(2B_a/B_H)**2) - alpha * B_a
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
        ## dvar / dt = 2 * var / (sqrt(pi) * bt) * (4 * A * Na * sigma - sqrt(pi) * bt * (A*Na + alpha + beta))
        ## dvar /dt = 2 * var * (-A*Na + 4A*Na*sigma/spi/bt + 6*var/bh**2 - 32*sigma**3/spi/bt/bh**2 - alpha - beta)
        spi = np.sqrt(np.pi)
        ts = torch.arange(0,self.T-1)
        alpha = self.return_alpha(ts)
        a = self.return_A(ts)
        bt = self.B_T[0]
        vars = self.sig[:-1]
        dvdt = (self.sig[1:]-self.sig[:-1])
        dvdt1 = vars * (-a + 4*a*vars/spi/bt - alpha) 
        return ((dvdt-dvdt1)).square().mean()/self.ce2
    
    def clip_params(self):
        with torch.no_grad():
            torch.clip_(self.A,0,100)
            torch.clip_(self.B_T,0.01,2)
            torch.clip_(self.B_H,0.1,10)
            torch.clip_(self.bas,0.1,0.6)
            torch.clip_(self.sig,0.01,0.3)

    def train_epoch(self,optimizer,lam1,lam2,lam3,lam4):
        optimizer.zero_grad()
        ld1 = self.L_d1()
        ld2 = self.L_d2()
        le1 = self.L_e1()
        le2 = self.L_e2()
        loss = lam1*ld1 + lam2*ld2 + lam3*le1 + lam4*le2 
        loss.backward()
        optimizer.step()
        self.clip_params()
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