import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

import scipy.stats

cuda = torch.cuda.device(0)

class SchrodingerBridge(nn.Module):

    def __init__(self):

        super(SchrodingerBridge,self).__init__()

        self.embed_num  = 15
        self.output_num = 1
        self.val_num    = 25000

        self.m = 1000
        self.n = 1000

        self.Bx = nn.Linear(self.embed_num,self.output_num,bias=True)
        self.By = nn.Linear(self.embed_num,self.output_num,bias=True)

        self.softplus = torch.nn.Softplus(beta=1,threshold=20)

    def Activation(self,x):

        return self.softplus(x) + 1e-8

    def LossX(self,x,y,x_,y_,p):

        X_i = self.Activation( self.Bx(x) )
        Y_j = self.Activation( self.By(y) )

        X_i_ = self.Activation( self.Bx(x_) )

        Y_j_ = self.Activation( F.conv1d(y_,self.By.weight.resize([self.output_num,self.embed_num,1])) )

        X = - (1/self.val_num)*sum(torch.log(X_i))
        C = (1/self.m)*(1/self.n)*sum((Y_j_.sum(2))*(X_i_/p))

        return X+C

    def LossY(self,x,y,x_,y_,p):

        X_i = self.softplus( self.Bx(x) ) + 1e-8
        Y_j = self.softplus( self.By(y) ) + 1e-8

        X_i_ = self.softplus( self.Bx(x_) ) + 1e-8

        Y_j_ = self.softplus( F.conv1d(y_,self.By.weight.resize([self.output_num,self.embed_num,1])) ) + 1e-8

        Y = - (1/self.val_num)*sum(torch.log(Y_j))
        C = (1/self.m)*(1/self.n)*sum((Y_j_.sum(2))*(X_i_/p))

        return Y+C

class GenerateValues:

    def __init__(self):

        super(GenerateValues,self).__init__()

        self.X = np.linspace(-15,15,1000)
        self.dX = np.mean(np.diff(self.X))

        self.fx_i = scipy.stats.norm.pdf(self.X,loc=0,scale=1)/sum(scipy.stats.norm.pdf(self.X,loc=0,scale=1))
        self.fy_j = scipy.stats.norm.pdf(self.X,loc=1,scale=3)/sum(scipy.stats.norm.pdf(self.X,loc=1,scale=3))

        self.x0 = self.Sample(self.X,self.fx_i)
        self.y0 = self.Sample(self.X,self.fy_j)

        self.x_i = self.x0[0][np.random.randint(len(self.x0[0]),size=[25000,])]
        self.y_j = self.y0[0][np.random.randint(len(self.y0[0]),size=[25000,])]

        self.fx_i_ = sum(np.exp(-np.power(self.X.reshape([1,len(self.X)])-self.x_i.reshape([len(self.x_i),1]),2)))

        self.x0_ = self.Sample(self.X,self.fx_i_)

        R = np.random.randint(len(self.x0_[0]),size=[1000,])

        self.x_i_ = self.x0_[0][R]
        self.p_x_i = self.x0_[1][R]

        self.y_j_ = np.ones([1000,1,1000])

        self.lengths = np.zeros([len(self.x_i_),1])

        for ii in range(0,len(self.x_i_)-1):

            self.y0_ = self.Sample(self.X,self.TransitionFunction(self.X,self.x0_[0][ii]))
            self.y0_ = self.y0_[0][np.random.randint(len(self.y0_[0]),size=[1000,])]
            self.y_j_[:,0,ii] = self.y0_

        self.Y = torch.Tensor(self.A(self.X))

        self.makeFunctionSpace()
        self.UseCuda()

    def TransitionFunction(self,x,y):

        return np.exp(-np.power(x-y,2))/sum(np.exp(-np.power(x-y,2)))

    def A(self,x):

        self.fXnum = 15

        y = np.ones([len(x),self.fXnum])

        y[:,0] = x * np.exp(-np.power(x,2))
        y[:,1] = np.power(x,2) * np.exp(-np.power(x,2))
        y[:,2] = np.power(x,3) * np.exp(-np.power(x,2))
        y[:,3] = np.power(x,4) * np.exp(-np.power(x,2))
        y[:,4] = np.power(x,5) * np.exp(-np.power(x,2))

        y[:,5] = x * np.exp(-np.power(x,2)/2)
        y[:,6] = np.power(x,2) * np.exp(-np.power(x,2)/5)
        y[:,7] = np.power(x,3) * np.exp(-np.power(x,2)/5)
        y[:,8] = np.power(x,4) * np.exp(-np.power(x,2)/5)
        y[:,9] = np.power(x,5) * np.exp(-np.power(x,2)/5)

        y[:,10] = x * np.exp(-np.power(x,2)/3)
        y[:,11] = np.power(x,2) * np.exp(-np.power(x,2)/10)
        y[:,12] = np.power(x,3) * np.exp(-np.power(x,2)/10)
        y[:,13] = np.power(x,4) * np.exp(-np.power(x,2)/10)
        y[:,14] = np.power(x,5) * np.exp(-np.power(x,2)/10)

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

    def makeFunctionSpace(self):

        self.X_i = torch.Tensor(self.A(self.x_i))
        self.Y_j = torch.Tensor(self.A(self.y_j))

        self.X_i_ = torch.Tensor(self.A(self.x_i_))
        self.Y_j_ = np.ones([1000,self.fXnum,len(self.x_i_)])

        for ii in range(1,len(self.x_i_)):

            self.Y_j_[:,:,ii] = self.A(self.y_j_[:,0,ii].reshape([1000]))

        self.Y_j_ = torch.Tensor(self.Y_j_).transpose(0,2)
        self.P = torch.Tensor([self.p_x_i])

    def UseCuda(self):

        self.X_i = self.X_i.cuda()
        self.Y_j = self.Y_j.cuda()

        self.X_i_ = self.X_i_.cuda()
        self.Y_j_ = self.Y_j_.cuda()

        self.P = self.P.cuda()

G = GenerateValues()

model = SchrodingerBridge()
model = model.cuda()

optimizerX = optim.SGD(model.parameters(),lr=1e-2,momentum=1e-3,nesterov=True,weight_decay=0)
optimizerY = optim.SGD(model.parameters(),lr=1e-2,momentum=1e-3,nesterov=True,weight_decay=0)

while True:

    optimizerX.zero_grad()

    lossX = model.LossX(G.X_i,G.Y_j,G.X_i_,G.Y_j_,G.P)
    lossX.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1e3)

    optimizerX.step()

    optimizerY.zero_grad()

    lossY = model.LossY(G.X_i,G.Y_j,G.X_i_,G.Y_j_,G.P)
    lossY.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1e3)

    optimizerY.step()

    if np.mod(epoch,100) == 0:

        """y = model.Activation( model.By(G.Y) )
        y = y.resize([1000])

        fx_i = torch.Tensor(G.fx_i)
        fy_j = torch.Tensor(G.fy_j)

        Y = torch.zeros(1000,1)

        for ii in range(len(fx_i)):

            Y = Y + ((fx_i[ii])*(y*torch.Tensor(G.TransitionFunction(G.X,G.X[ii])))/sum(y*torch.Tensor(G.TransitionFunction(G.X,G.X[ii])))).resize([1000,1])

        Y = Y.data.numpy()

        Y = Y.resize([1000])

        print(sum(Y*np.log((Y+1e-8)/(G.fy_j+1e-8))))"""

        print(lossX.data.numpy())
        print(lossY.data.numpy())

        torch.save(model.state_dict(), PATH)
