
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

import scipy.stats

import matplotlib.pyplot as plt


# In[442]:


model = SchrodingerBridge()

Lx = model.softplus( model.Bx3( model.Bx2( model.Bx1(G.Y) ) ) )  + 1e-8
Ly = model.softplus( model.Bx3( model.Bx2( model.Bx1(G.Y) ) ) )  + 1e-8

#LX = - torch.log(Lx * torch.FloatTensor(vals.fx_i)).sum() * vals.dX

Lx*torch.FloatTensor(G.fx_i)


# In[467]:


class SchrodingerBridge(nn.Module):
    
    def __init__(self):
        
        super(SchrodingerBridge,self).__init__()
        
        self.embed_num1    = 25
        self.embed_num2    = 1
        self.embed_num3    = 5
        
        self.output_num = 1
        self.val_num    = 1000
        
        self.m = self.val_num
        self.n = 100
        
        self.Bx1 = nn.Linear(self.embed_num1,self.embed_num2,bias=True)
        self.By1 = nn.Linear(self.embed_num1,self.embed_num2,bias=True)
        
        self.Bx2 = nn.Linear(self.embed_num2,self.embed_num3,bias=True)
        self.By2 = nn.Linear(self.embed_num2,self.embed_num3,bias=True)

        self.Bx3 = nn.Linear(self.embed_num3,self.output_num,bias=True)
        self.By3 = nn.Linear(self.embed_num3,self.output_num,bias=True)

        self.softplus = torch.nn.Softplus(beta=10,threshold=20)
        self.ReLU = torch.nn.LeakyReLU(negative_slope=0.5)
        
    def Activation(self,x):
        
        return self.softplus(x) + 1e-8
        
    def LossX(self,x,y,x_,y_,p):
        
        """X_i = self.softplus( self.Bx3( self.ReLU( self.Bx2( self.ReLU( self.Bx1(x) ) ) ) ) ) + 1e-8
        Y_j = self.softplus( self.Bx3( self.ReLU( self.Bx2( self.ReLU( self.Bx1(y) ) ) ) ) ) + 1e-8
        
        X_i_ = self.softplus( self.Bx3( self.ReLU( self.Bx2( self.ReLU( self.Bx1(x_) ) ) ) ) ) + 1e-8
        
        y1 = self.ReLU( F.conv1d(y_,self.By1.weight.reshape([self.embed_num2,self.embed_num1,1])) )
        y2 = self.ReLU( F.conv1d(y1,self.By2.weight.reshape([self.embed_num3,self.embed_num2,1])) )
        Y_j_ = self.softplus( F.conv1d(y2,self.By3.weight.reshape([self.output_num,self.embed_num3,1])) ) + 1e-8"""
        
        X_i = self.softplus( self.Bx3( self.Bx2( self.Bx1(x) ) ) )  + 1e-8
        Y_j = self.softplus( self.By3( self.By2( self.By1(y) ) ) )  + 1e-8
        
        X_i_ = self.softplus( self.Bx3( self.Bx2( self.Bx1(x_) ) ) ) + 1e-8
        
        y1 = F.conv1d(y_,self.By1.weight.reshape([self.embed_num2,self.embed_num1,1]))
        y2 = F.conv1d(y1,self.By2.weight.reshape([self.embed_num3,self.embed_num2,1]))
        Y_j_ = self.softplus( F.conv1d(y2,self.By3.weight.reshape([self.output_num,self.embed_num3,1])) ) + 1e-8
        
        """X_i = self.softplus( self.Bx1(x) ) + 1e-8
        Y_j = self.softplus( self.Bx1(y) ) + 1e-8
        
        X_i_ = self.softplus( self.Bx1(x_) ) + 1e-8
        
        Y_j_ = self.softplus( F.conv1d(y_,self.By1.weight.reshape([self.embed_num2,self.embed_num1,1])) ) + 1e-8"""
        
        X = - (1/self.val_num)*sum(torch.log(X_i))
        C = (1/self.m)*(1/self.n)*sum((Y_j_.sum(2))*(X_i_/p))
        return X+C
    
    def LossY(self,x,y,x_,y_,p):
        
        """X_i = self.softplus( self.Bx3( self.ReLU( self.Bx2( self.ReLU( self.Bx1(x) ) ) ) ) ) + 1e-8
        Y_j = self.softplus( self.By3( self.ReLU( self.By2( self.ReLU( self.By1(y) ) ) ) ) ) + 1e-8
        
        X_i_ = self.softplus( self.Bx3( self.ReLU( self.Bx2( self.ReLU( self.Bx1(x_) ) ) ) ) ) + 1e-8
        
        y1 = self.ReLU( F.conv1d(y_,self.By1.weight.reshape([self.embed_num2,self.embed_num1,1])) )
        y2 = self.ReLU( F.conv1d(y1,self.By2.weight.reshape([self.embed_num3,self.embed_num2,1])) )
        Y_j_ = self.softplus( F.conv1d(y2,self.By3.weight.reshape([self.output_num,self.embed_num3,1])) ) + 1e-8"""
        
        X_i = self.softplus( self.Bx3( self.Bx2( self.Bx1(x) ) ) )  + 1e-8
        Y_j = self.softplus( self.By3( self.By2( self.By1(y) ) ) )  + 1e-8
        
        X_i_ = self.softplus( self.Bx3( self.Bx2( self.Bx1(x_) ) ) ) + 1e-8
        
        y1 = F.conv1d(y_,self.By1.weight.reshape([self.embed_num2,self.embed_num1,1]))
        y2 = F.conv1d(y1,self.By2.weight.reshape([self.embed_num3,self.embed_num2,1]))
        Y_j_ = self.softplus( F.conv1d(y2,self.By3.weight.reshape([self.output_num,self.embed_num3,1])) ) + 1e-8
        
        """X_i = self.softplus( self.Bx1(x) ) + 1e-8
        Y_j = self.softplus( self.Bx1(y) ) + 1e-8
        
        X_i_ = self.softplus( self.Bx1(x_) ) + 1e-8
        
        Y_j_ = self.softplus( F.conv1d(y_,self.By1.weight.reshape([self.embed_num2,self.embed_num1,1])) ) + 1e-8"""
        
        Y = - (1/self.val_num)*sum(torch.log(Y_j))
        C = (1/self.m)*(1/self.n)*sum((Y_j_.sum(2))*(X_i_/p))
        
        return Y+C
        
    def ProbableLossX(self,vals):
        
        #Lx = self.softplus( self.Bx3( self.Bx2( self.Bx1(vals.Y) ) ) )  + 1e-8
        #Ly = self.softplus( self.By3( self.By2( self.By1(vals.Y) ) ) )  + 1e-8
        
        Lx = self.softplus( self.Bx1(vals.Y) ) + 1e-8
        Ly = self.softplus( self.By1(vals.Y) ) + 1e-8
                
        LX = - torch.log(Lx * torch.FloatTensor(vals.fx_i) + 1e-8).sum() * vals.dX
        LXY = (torch.conv1d(Ly.reshape([1000,1,1]),G.TransitionFunction(torch.FloatTensor(G.X),0).reshape([1000,1,1])).sum(1)*Lx).sum() * vals.dX
        return LX + LXY
    
    def ProbableLossY(self,vals):
        
        #Lx = self.softplus( self.Bx3( self.Bx2( self.Bx1(vals.Y) ) ) )  + 1e-8
        #Ly = self.softplus( self.By3( self.By2( self.By1(vals.Y) ) ) )  + 1e-8
        
        Lx = self.softplus( self.Bx1(vals.Y) ) + 1e-8
        Ly = self.softplus( self.By1(vals.Y) ) + 1e-8
        
        LY = - torch.log(Ly * torch.FloatTensor(vals.fy_j) + 1e-8).sum() * vals.dX
        LXY = (torch.conv1d(Ly.reshape([1000,1,1]),G.TransitionFunction(torch.FloatTensor(G.X),0).reshape([1000,1,1])).sum(1)*Lx).sum() * vals.dX
        return LY + LXY


# In[381]:


class GenerateValues:
    
    def __init__(self):
        
        super(GenerateValues,self).__init__()
        
        self.fXnum = 25
        
        self.X = np.linspace(-10,15,1000)
        self.dX = np.mean(np.diff(self.X))
        
        self.fx_i = scipy.stats.norm.pdf(self.X,loc=0,scale=1)/sum(scipy.stats.norm.pdf(self.X,loc=0,scale=1))
        self.fy_j = scipy.stats.norm.pdf(self.X,loc=1,scale=3)/sum(scipy.stats.norm.pdf(self.X,loc=1,scale=3))
        
        self.x0 = self.Sample(self.X,self.fx_i)
        self.y0 = self.Sample(self.X,self.fy_j)
        
        self.X0 = self.x0[0][np.random.randint(len(self.x0[0]),size=[1000,])]
        self.Y0 = self.y0[0][np.random.randint(len(self.y0[0]),size=[1000,])]
        
        self.x_i = self.A(self.X0)
        self.y_j = self.A(self.Y0)
        
        self.fx_i_ = sum(np.exp(-np.power(self.X.reshape([1,len(self.X)])-self.X0.reshape([len(self.X0),1]),2)))
        
        self.x0_ = self.Sample(self.X,self.fx_i_)
        
        R = np.random.randint(len(self.x0_[0]),size=[1000,])
        
        self.x_i_ = self.A(self.x0_[0][R])
        self.p_x_i = self.x0_[1][R]
        
        self.y_j_ = np.ones([100,self.fXnum,len(self.x_i_)])
        
        for ii in range(1,len(self.x_i_)):
            
            self.y0_ = self.Sample(self.X,self.TransitionFunction(self.X,self.x0_[0][ii]))
            self.y0_ = self.y0_[0][np.random.randint(len(self.y0_[0]),size=[100,])]
            self.y_j_[:,:,ii] = self.A(self.y0_).transpose(0,1)
        
        self.X_i = torch.Tensor(self.x_i)
        self.Y_j = torch.Tensor(self.y_j)
        
        #self.X_i = self.X_i.reshape([1000,self.fXnum,1])
        #self.Y_j = self.Y_j.reshape([1000,self.fXnum,1])
        
        self.X_i_ = torch.Tensor(self.x_i_)
        #self.X_i_ = self.X_i_.reshape([1000,self.fXnum,1])
        
        self.Y_j_ = torch.Tensor(self.y_j_)
        self.Y_j_ = self.Y_j_.transpose(0,2)
        
        self.P = torch.Tensor(self.p_x_i)
        self.P = self.P.reshape([1000,1])
        
        self.Y = torch.Tensor(self.A(self.X))
        
    def TransitionFunction(self,x,y):
        
        return np.exp(-np.power(x-y,2))/sum(np.exp(-np.power(x,2)))
        
    def A(self,x):
        
        y = np.ones([len(x),self.fXnum])
        
        y[:,0] = (2 * x)
        y[:,1] = (4 * np.power(x,2)) - 2
        y[:,2] = (8 * np.power(x,3)) - (12 * x)
        y[:,3] = (16 * np.power(x,4)) - (48 * np.power(x,2)) + 12
        y[:,4] = (32 * np.power(x,5)) - (160 * np.power(x,3)) + (120 * x)
        y[:,5] = (64 * np.power(x,5)) - (480 * np.power(x,3)) + (720 * np.power(x,2)) - 120
        y[:,6] = (128 * np.power(x,7)) - (1344 * np.power(x,5)) + (3360 * np.power(x,3)) - 1680
        y[:,7] = (256 * np.power(x,8)) - (3584 * np.power(x,6)) + (13440 * np.power(x,4)) - (13440 * np.power(x,2)) + 1680
        y[:,8] = (512 * np.power(x,9)) - (9216 * np.power(x,7)) + (48384 * np.power(x,5)) - (80640 * np.power(x,3)) + 30240 * x
        y[:,9] = (1024 * np.power(x,10)) - (23040 * np.power(x,8)) + (161280 * np.power(x,6)) - (403200 * np.power(x,4)) + (302400 * np.power(x,2)) - 30240
                
        y[:,10] = x * np.exp(-np.power(x,2))
        y[:,11] = np.power(x,2) * np.exp(-np.power(x,2))
        y[:,12] = np.power(x,3) * np.exp(-np.power(x,2))
        y[:,13] = np.power(x,4) * np.exp(-np.power(x,2))
        y[:,14] = np.power(x,5) * np.exp(-np.power(x,2))
        
        y[:,15] = x * np.exp(-np.power(x,2)/2)
        y[:,16] = np.power(x,2) * np.exp(-np.power(x,2)/2)
        y[:,17] = np.power(x,3) * np.exp(-np.power(x,2)/2)
        y[:,18] = np.power(x,4) * np.exp(-np.power(x,2)/2)
        y[:,19] = np.power(x,5) * np.exp(-np.power(x,2)/2)
        
        y[:,20] = x * np.exp(-np.power(x,2)/3)
        y[:,21] = np.power(x,2) * np.exp(-np.power(x,2)/3)
        y[:,22] = np.power(x,3) * np.exp(-np.power(x,2)/3)
        y[:,23] = np.power(x,4) * np.exp(-np.power(x,2)/3)
        y[:,24] = np.power(x,5) * np.exp(-np.power(x,2)/3)
        
        return y
        
    def Sample(self,X,Y):
        
        x_i = np.array([])
        p_x_i = np.array([])

        freq = np.ceil(10/(X[1]-X[0]))
        fY = (Y-min(Y))/(max(Y)-min(Y))
        pY = Y/sum(Y)

        for ii in range(1,len(X)):

            x_i = np.append(x_i,(X[ii]+np.random.randn(int(np.ceil(freq*fY[ii])),1))*np.ones([int(np.ceil(freq*fY[ii])),1]))
            p_x_i = np.append(p_x_i,(pY[ii])*np.ones([int(np.ceil(freq*fY[ii])),1]))

        return x_i,p_x_i
    
    """def MovetoCuda(self,cuda):
        
        self.X_i = self.X_i.to(device=cuda)
        self.Y_j = self.Y_j.to(device=cuda)
        
        self.X_i_ = self.X_i_.to(device=cuda)
        self.Y_j_ = self.Y_j_.to(device=cuda)
        
        self.P = self.P.to(device=cuda)"""
        


# In[382]:


G = GenerateValues()


# In[468]:


model = SchrodingerBridge()


# In[470]:


optimizer = optim.Adamax(model.parameters(),lr=1e-2)

for epoch in range(1000000):
    
    optimizer.zero_grad()
    
    lossX = model.LossX(G.X_i,G.Y_j,G.X_i_,G.Y_j_,G.P)
    #lossX = model.ProbableLossX(G)
    lossX.backward()
    optimizer.step()
    
    optimizer.zero_grad()
    
    lossY = model.LossY(G.X_i,G.Y_j,G.X_i_,G.Y_j_,G.P)
    #lossY = model.ProbableLossY(G)
    lossY.backward()
    optimizer.step()
    
    if np.mod(epoch,10) == 0:
              
        y = model.softplus( model.Bx3( model.softplus( model.Bx2( model.softplus( model.Bx1(G.Y) ) ) ) ) ) + 1e-8
        y = y.reshape([1000])
        y = y.data.numpy()

        Y = y*G.TransitionFunction(G.X,0)/sum(y*G.TransitionFunction(G.X,0))

        print(sum(Y*np.log((Y+1e-8)/(G.fy_j+1e-8))))
        
        #torch.save(model,'checkpoint.pth')
        
        print(lossX.data.numpy())
        print(lossY.data.numpy())


# In[465]:


y = model.softplus( model.Bx3( model.softplus( model.Bx2( model.softplus( model.Bx1(G.Y) ) ) ) ) ) + 1e-8
y = y.reshape([1000])
y = y.data.numpy()

Y = y*G.TransitionFunction(G.X,0)/sum(y*G.TransitionFunction(G.X,0))
Y = np.convolve(Y,G.fx_i,"same")

plt.plot(G.X,Y)
plt.plot(G.X,G.fy_j)
plt.show()


# In[466]:


plt.plot(G.X,G.fx_i)
plt.plot(G.X,G.fy_j)
plt.show()


# In[35]:


torch.save(model,'checkpoint.pth')


# In[37]:


model = torch.load('checkpoint.pth')


# model

# In[38]:


model


# In[357]:


model.state_dict()


# In[117]:


model.state_dict()


# In[211]:


G.Y_j_.size()
y1 = F.conv1d(G.Y_j_,model.By1.weight.reshape([model.embed_num2,model.embed_num1,1]))
y2 = F.conv1d(y1,model.By2.weight.reshape([model.embed_num3,model.embed_num2,1]))
y3 = F.conv1d(y2,model.By3.weight.reshape([model.output_num,model.embed_num3,1]))

#F.conv1d(Y_j_,model.By.weight.reshape([1,self.function_num,1])


# In[209]:


model.By2.weight.size()


# In[212]:


y3.size()

