import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

import scipy.stats

class SchrodingerBridge(nn.Module):

    def __init__(self):

        super(SchrodingerBridge,self).__init__()

        self.embed_num    = 31
        self.function_num = 10

        self.output_num = 1
        self.val_num    = 1000

        self.m = self.val_num
        self.n = 10

        self.BxC = nn.Conv1d(self.embed_num, self.function_num, 1, stride=5,bias=True)
        self.ByC = nn.Conv1d(self.embed_num, self.function_num, 1, stride=5,bias=True)

        self.Bx = nn.Linear(self.function_num,self.output_num,bias=True)
        self.By = nn.Linear(self.function_num,self.output_num,bias=True)

        self.Bx.weight.data = torch.zeros(1,self.function_num)
        self.By.weight.data = torch.zeros(1,self.function_num)

        self.softplus = torch.nn.Softplus(beta=1,threshold=20)

    def Activation(self,x):

        return self.softplus(x) + 1e-8

    def LossX(self,x,y,x_,y_,p):

        X_i = self.BxC(x)
        Y_j = self.ByC(y)

        X_i_ = self.BxC(x_)
        Y_j_ = self.ByC(y_)

        X_i = X_i.reshape([self.val_num,self.function_num])
        Y_j = Y_j.reshape([self.val_num,self.function_num])

        X_i_ = X_i_.reshape([self.val_num,self.function_num])

        X = - (1/self.val_num)*sum(torch.log(self.Activation(self.Bx(X_i))))
        C = (1/self.m)*(1/self.n)*sum(self.Activation(F.conv1d(Y_j_,self.By.weight.reshape([1,self.function_num,1]))).sum(2)*self.Activation(self.Bx(X_i_)/p))
        return X+C

    def LossY(self,x,y,x_,y_,p):

        X_i = self.BxC(x)
        Y_j = self.ByC(y)

        X_i_ = self.BxC(x_)
        Y_j_ = self.ByC(y_)

        X_i = X_i.reshape([self.val_num,self.function_num])
        Y_j = Y_j.reshape([self.val_num,self.function_num])

        X_i_ = X_i_.reshape([self.val_num,self.function_num])

        Y = - (1/self.val_num)*sum(torch.log(self.Activation(self.By(Y_j))))
        C = (1/self.m)*(1/self.n)*sum(self.Activation(F.conv1d(Y_j_,self.By.weight.reshape([1,self.function_num,1]))).sum(2)*self.Activation(self.Bx(X_i_)/p))
        return Y+C

class GenerateValues:

    def __init__(self):

        super(GenerateValues,self).__init__()

        self.fXnum = 31

        self.X = np.linspace(-10,15,1000)

        self.fx_i = scipy.stats.lognorm.pdf(self.X,1,loc=0,scale=1)/sum(scipy.stats.lognorm.pdf(self.X,1,loc=0,scale=1))
        self.fy_j = scipy.stats.lognorm.pdf(self.X,1,loc=0,scale=3)/sum(scipy.stats.lognorm.pdf(self.X,1,loc=0,scale=3))

        self.x0 = self.Sample(self.X,self.fx_i)
        self.y0 = self.Sample(self.X,self.fy_j)

        self.x0 = self.x0[0][np.random.randint(len(self.x0[0]),size=[1000,])]
        self.y0 = self.y0[0][np.random.randint(len(self.y0[0]),size=[1000,])]

        self.x_i = self.A(self.x0)
        self.y_j = self.A(self.y0)

        fx_i_ = sum(np.exp(-np.power(self.X.reshape([1,len(self.X)])-self.x0.reshape([len(self.x0),1]),2)))

        self.x0_ = self.Sample(self.X,fx_i_)

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

        self.X_i = self.X_i.reshape([1000,self.fXnum,1])
        self.Y_j = self.Y_j.reshape([1000,self.fXnum,1])

        self.X_i_ = torch.Tensor(self.x_i_)
        self.X_i_ = self.X_i_.reshape([1000,self.fXnum,1])

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

        y[:,10] = np.exp(- 3 * x)
        y[:,11] = np.exp(- 2 * x)
        y[:,12] = np.exp(- 1 * x)
        y[:,13] = np.exp(1 * x)
        y[:,14] = np.exp(2 * x)
        y[:,15] = np.exp(3 * x)

        y[:,16] = x * np.exp(-np.power(x,2))
        y[:,17] = np.power(x,2) * np.exp(-np.power(x,2))
        y[:,18] = np.power(x,3) * np.exp(-np.power(x,2))
        y[:,19] = np.power(x,4) * np.exp(-np.power(x,2))
        y[:,20] = np.power(x,5) * np.exp(-np.power(x,2))

        y[:,21] = x * np.exp(-np.power(x,2)/2)
        y[:,22] = np.power(x,2) * np.exp(-np.power(x,2)/2)
        y[:,23] = np.power(x,3) * np.exp(-np.power(x,2)/2)
        y[:,24] = np.power(x,4) * np.exp(-np.power(x,2)/2)
        y[:,25] = np.power(x,5) * np.exp(-np.power(x,2)/2)

        y[:,26] = x * np.exp(-np.power(x,2)/3)
        y[:,27] = np.power(x,2) * np.exp(-np.power(x,2)/3)
        y[:,28] = np.power(x,3) * np.exp(-np.power(x,2)/3)
        y[:,29] = np.power(x,4) * np.exp(-np.power(x,2)/3)
        y[:,30] = np.power(x,5) * np.exp(-np.power(x,2)/3)

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

G = GenerateValues()

model = SchrodingerBridge()
optimizer = optim.Adadelta(model.parameters(), lr = 1e-1)

for epoch in range(1000000):

    optimizer.zero_grad()

    lossX = model.LossX(G.X_i,G.Y_j,G.X_i_,G.Y_j_,G.P)
    lossX.backward()
    optimizer.step()

    optimizer.zero_grad()

    lossY = model.LossY(G.X_i,G.Y_j,G.X_i_,G.Y_j_,G.P)
    lossY.backward()
    optimizer.step()

    if np.mod(epoch,100) == 0:

        Y = model.ByC(torch.tensor(G.Y).reshape([1000,model.embed_num,1]))
        Y = Y.reshape([1000,model.function_num])
        Y = model.Activation(model.By(Y))
        Y = Y.data.numpy()
        Y = np.convolve(G.fx_i,Y.reshape([1000])*G.TransitionFunction(G.X,0)/sum(Y.reshape([1000])*G.TransitionFunction(G.X,0)),"same")

        torch.save(model.state_dict(), PATH)

        print(sum(Y*np.log((Y+1e-8)/(G.fy_j+1e-8))))
        print(lossX.data.numpy())
        print(lossY.data.numpy())
