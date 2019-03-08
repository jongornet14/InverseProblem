import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os

class AutoEncoderBridgeConv(nn.Module):

    def __init__(self):

        super(AutoEncoderBridgeConv,self).__init__()

        self.image_size = 784
        self.hidden_size = 50
        self.dist_size = 20

        self.batch_size = 1000
        self.fnum = 15

        self.Lencode1 = nn.Conv2d(1,self.dist_size,10)
        self.Lencode2 = nn.Conv2d(self.dist_size,self.hidden_size,10)

        self.EncodePool = linear = nn.Linear(5000,self.dist_size)

        self.Ldecode1 = nn.ConvTranspose2d(self.hidden_size,self.dist_size,10)
        self.Ldecode2 = nn.ConvTranspose2d(self.dist_size,1,10)

        self.DecodePool = nn.Linear(self.dist_size,5000)

        self.bin_num = 300

        self.probspace = torch.linspace(-100,100,steps=self.bin_num).reshape([1,self.bin_num]).cuda()
        self.numpy_probspace = np.linspace(-100,100,self.bin_num).reshape([1,self.bin_num])

    def Encode(self,x):

        x = self.Lencode1(x)
        x = self.Lencode2(x).view(self.batch_size,50*10*10)
        x = self.EncodePool(x).transpose(0,1)

        return x

    def Decode(self,x):

        x = self.DecodePool(x)
        x = self.Ldecode1(x.view(self.batch_size,50,10,10))
        x = self.Ldecode2(x)

        return x

    def InverseDecode(self,x):

        x = F.conv2d(x,self.Ldecode2.weight.data)
        x = F.conv2d(x,self.Ldecode1.weight.data).view(self.batch_size,50*10*10)
        x = torch.mm(x,self.DecodePool.weight.data).transpose(0,1)

        return x

    def Bridge(self,x):

        x_i = self.Encode(x)
        y_j = self.InverseDecode(x)

        f_x = (-(self.probspace.reshape([1,self.bin_num,1])-x_i.reshape([self.dist_size,1,self.batch_size])).pow(2)).exp()
        f_x = f_x.sum(2)
        f_x = f_x/f_x.sum(1).reshape([self.dist_size,1])
        f_x = f_x.cumsum(1)

        r = torch.rand([self.dist_size,model.bin_num,self.batch_size]).cuda()
        pos = (f_x.reshape([self.dist_size,self.bin_num,1])-r).abs().min(1)

        P = pos[0]
        positions = pos[1]

        x_i_ = self.probspace.reshape([self.bin_num,1,1])[positions].squeeze()
        prob = torch.distributions.normal.Normal(x_i_, 1)
        y_j_ = prob.sample(sample_shape=torch.Size([10,1])).transpose(0,2).transpose(1,3).transpose(1,2).squeeze()

        self.x_i = x_i
        self.y_j = y_j
        self.x_i_ = x_i_
        self.y_j_ = y_j_

        self.P = P.reshape([self.dist_size,self.batch_size,1])

    def Forward(self,x,bridgemodels):

        out = self.Encode(x)
        vals = torch.zeros([self.batch_size,self.dist_size]).cuda()
        for index in range(self.dist_size):
            bvalue = bridgemodels[index].GenerateValues(self.probspace,self.x_i[index,:])
            vals[:,index] = bvalue[1]
        out = self.Decode(vals)

        return out

class Bridge(nn.Module):

    def __init__(self):

        super(Bridge,self).__init__()

        self.dist_num = 20
        self.bin_num = 300

        self.mn = 1000

        self.m_ = 1000
        self.n_ = 10

        self.input_num = 15
        self.output_num = 1

        self.Softplus = nn.Softplus()

        self.Bx = nn.Linear(self.input_num,self.output_num)
        self.By = nn.Linear(self.input_num,self.output_num)

    def Aspace(self,x):

        y = torch.ones([self.bin_num,self.input_num]).cuda()

        y[:,0] = (-x.pow(2)).exp()
        y[:,1] = (x)*(-x.pow(2)).exp()
        y[:,2] = (x.pow(2))*(-x.pow(2)).exp()
        y[:,3] = (x.pow(3))*(-x.pow(2)).exp()
        y[:,4] = (x.pow(4))*(-x.pow(2)).exp()

        y[:,5] = (-x.pow(2)/25).exp()
        y[:,6] = (x)*(-x.pow(2)/25).exp()
        y[:,7] = (x.pow(2))*(-x.pow(2)/25).exp()
        y[:,8] = (x.pow(3))*(-x.pow(2)/25).exp()
        y[:,9] = (x.pow(4))*(-x.pow(2)/25).exp()

        y[:,10] = (-x.pow(2)/50).exp()
        y[:,11] = (x)*(-x.pow(2)/50).exp()
        y[:,12] = (x.pow(2))*(-x.pow(2)/50).exp()
        y[:,13] = (x.pow(3))*(-x.pow(2)/50).exp()
        y[:,14] = (x.pow(4))*(-x.pow(2)/50).exp()

        return y

    def A(self,x):

        y = torch.ones([self.mn,self.input_num]).cuda()

        y[:,0] = (-x.pow(2)).exp()
        y[:,1] = (x)*(-x.pow(2)).exp()
        y[:,2] = (x.pow(2))*(-x.pow(2)).exp()
        y[:,3] = (x.pow(3))*(-x.pow(2)).exp()
        y[:,4] = (x.pow(4))*(-x.pow(2)).exp()

        y[:,5] = (-x.pow(2)/25).exp()
        y[:,6] = (x)*(-x.pow(2)/25).exp()
        y[:,7] = (x.pow(2))*(-x.pow(2)/25).exp()
        y[:,8] = (x.pow(3))*(-x.pow(2)/25).exp()
        y[:,9] = (x.pow(4))*(-x.pow(2)/25).exp()

        y[:,10] = (-x.pow(2)/50).exp()
        y[:,11] = (x)*(-x.pow(2)/50).exp()
        y[:,12] = (x.pow(2))*(-x.pow(2)/50).exp()
        y[:,13] = (x.pow(3))*(-x.pow(2)/50).exp()
        y[:,14] = (x.pow(4))*(-x.pow(2)/50).exp()

        return y

    def AConv(self,x):

        y = torch.ones([self.n_,self.m_,self.input_num]).cuda()

        y[:,:,0] = (-x.pow(2)).exp()
        y[:,:,1] = (x)*(-x.pow(2)).exp()
        y[:,:,2] = (x.pow(2))*(-x.pow(2)).exp()
        y[:,:,3] = (x.pow(3))*(-x.pow(2)).exp()
        y[:,:,4] = (x.pow(4))*(-x.pow(2)).exp()

        y[:,:,5] = (-x.pow(2)/25).exp()
        y[:,:,6] = (x)*(-x.pow(2)/25).exp()
        y[:,:,7] = (x.pow(2))*(-x.pow(2)/25).exp()
        y[:,:,8] = (x.pow(3))*(-x.pow(2)/25).exp()
        y[:,:,9] = (x.pow(4))*(-x.pow(2)/25).exp()

        y[:,:,10] = (-x.pow(2)/50).exp()
        y[:,:,11] = (x)*(-x.pow(2)/50).exp()
        y[:,:,12] = (x.pow(2))*(-x.pow(2)/50).exp()
        y[:,:,13] = (x.pow(3))*(-x.pow(2)/50).exp()
        y[:,:,14] = (x.pow(4))*(-x.pow(2)/50).exp()

        return y

    def Activation(self,x):

        return self.Softplus(x) + 1e-8

    def LayerX(self,x):

        out = self.A(x)
        out = self.Bx(out)
        out = self.Activation(out)

        return out

    def LayerY(self,x):

        out = self.A(x)
        out = self.By(out)
        out = self.Activation(out)

        return out

    def ConvLayerY(self,x):

        out = self.AConv(x)
        out = self.By(out)
        out = self.Activation( out )

        return out

    def LayerPred(self,x):

        out = self.Aspace(x)
        out = self.By(out)
        out = self.Activation(out)

        return out

    def BridgeX(self,x_i,y_j,x_i_,y_j_,P):

        X_i = self.LayerX(x_i)
        X_i_ = self.LayerX(x_i_)

        Y_j_ = self.ConvLayerY(y_j_)

        X = - (1/self.mn)*torch.log(X_i).sum(0)
        C = (1/self.m_)*(1/self.n_)*(Y_j_.sum(0)*(X_i_/P)).sum(0)

        return X+C

    def BridgeY(self,x_i,y_j,x_i_,y_j_,P):

        X_i_ = self.LayerX(x_i_)

        Y_j = self.LayerY(y_j)
        Y_j_ = self.ConvLayerY(y_j_)

        Y = - (1/self.mn)*torch.log(Y_j).sum(0)
        C = (1/self.m_)*(1/self.n_)*(Y_j_.sum(0)*(X_i_/P)).sum(0)

        return Y+C

    def GenerateValues(self,probspace,x_i):

        y = self.LayerPred(probspace.squeeze())

        p = scipy.stats.norm.pdf(probspace.cpu().data.numpy().reshape([300,1]),loc=x_i.cpu().data.numpy().reshape([1,self.mn]),scale=1)
        p = torch.Tensor(p).cuda()

        pred = p*y
        pred = pred.sum(1)
        pred = pred/pred.sum()
        predPDF = pred
        pred = pred.cumsum(0).reshape([self.bin_num,1])

        r = torch.rand([1,self.mn,]).cuda()
        pos = (pred-r).abs().min(0)

        samples = probspace.squeeze()[pos[1]].squeeze()

        return predPDF,samples

class Distributions(nn.Module):

    def __init__(self):

        super(Distributions,self).__init__()

        self.x_i = torch.zeros([10,20,1000])
        self.y_j = torch.zeros([10,20,1000])
        self.x_i_ = torch.zeros([10,20,1000])
        self.y_j_ = torch.zeros([10,20,10,1000])

        self.P = torch.zeros([10,20,1000,1])

    def DevelopMemories(self,data,digit):

        self.x_i[digit,:,:] = data.x_i
        self.y_j[digit,:,:] = data.y_j
        self.x_i_[digit,:,:] = data.x_i_
        self.y_j_[digit,:,:,:] = data.y_j_

        self.P[digit,:,:,:] = data.P

    def removeGrad(self):

        self.x_i = self.x_i.detach()
        self.y_j = self.y_j.detach()
        self.x_i_ = self.x_i_.detach()
        self.y_j_ = self.y_j_.detach()

        self.P = self.P.detach()

    def movetoCuda(self):

        self.x_i = self.x_i.cuda()
        self.y_j = self.y_j.cuda()
        self.x_i_ = self.x_i_.cuda()
        self.y_j_ = self.y_j_.cuda()

        self.P = self.P.cuda()

def AutoEncoderLossFunction(num,epoch,torchimage,imageLoss):

    model.Bridge(torchimage)

    Loss = 0

    distributions.DevelopMemories(model,batch)
    distributions.removeGrad()

    recon_image = model.Forward(torchimage,bridges)
    recon_loss = F.smooth_l1_loss(recon_image,torchimage,reduction='mean')

    optimizerAutoencoder.zero_grad()
    recon_loss.backward()
    optimizerAutoencoder.step()

    imageLoss = (0.1*recon_loss.cpu().data.numpy()) + imageLoss

    if num == 9 & epoch == 9:

        print('Image Loss: ' + str(imageLoss))
        imageLoss = 0

def BridgeLossFunction(num,epoch,LmuX,LmuY):

    for b in range(model.dist_size):

        x_i = distributions.x_i[batch,b,:]
        y_j = distributions.y_j[batch,b,:]
        x_i_ = distributions.x_i_[batch,b,:]
        y_j_ = distributions.y_j_[batch,b,:,:]
        P = distributions.P[batch,b,:]

        bridges[b].Bx.weight.requires_grad = True
        bridges[b].Bx.bias.requires_grad = True

        bridges[b].By.weight.requires_grad = False
        bridges[b].By.bias.requires_grad = False

        LossX = bridges[b].BridgeX(x_i,y_j,x_i_,y_j_,P)

        optimizerBridges[b].zero_grad()
        LossX.backward()

        LmuX = ((1/(model.dist_size))*LossX.cpu().data.numpy()[0]) + LmuX

        optimizerBridges[b].step()

        bridges[b].Bx.weight.requires_grad = False
        bridges[b].Bx.bias.requires_grad = False

        bridges[b].By.weight.requires_grad = True
        bridges[b].By.bias.requires_grad = True

        LossY = bridges[b].BridgeY(x_i,y_j,x_i_,y_j_,P)

        optimizerBridges[b].zero_grad()
        LossY.backward()

        LmuY = ((1/(model.dist_size))*LossY.cpu().data.numpy()[0]) + LmuY

        optimizerBridges[b].step()

    if epoch == 999:

        print('Bridge Loss X: ' + str(LmuX))
        print('Bridge Loss Y: ' + str(LmuY))

        LmuX = 0
        LmuY = 0

def saveModels():

    PATH = '/home/jmg1030/Documents/AutoEncoders/encoder.pth'
    torch.save(model.state_dict(), PATH)

    for b in range(model.dist_size):

        PATH = '/home/jmg1030/Documents/AutoEncoders/bridge_num_' + str(b) + '.pth'
        bridgemodel = bridges[b]
        torch.save(bridgemodel.state_dict(), PATH)

def loadModels():

    PATH = '/home/jmg1030/Documents/AutoEncoders/encoder.pth'
    if os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH))
        model.eval()

    for b in range(model.dist_size):
        PATH = '/home/jmg1030/Documents/AutoEncoders/bridge_num_' + str(b) + '.pth'
        if os.path.exists(PATH):
            bridges[b].load_state_dict(torch.load(PATH))

def loadData():

    data = np.loadtxt(open('/home/jmg1030/Documents/AutoEncoders/mnist_train.csv', 'rb'), delimiter=',', skiprows=1)
    for ii in range(10):
        image = np.zeros([1000,785])
        num = 0
        for jj in range(60000):
            if data[jj,0] == ii:
                image[num,:] = data[jj,:]
                num += 1
            if num == 1000:
                images[ii,:,:] = image
                break

distributions = Distributions()

images = np.zeros([10,1000,785])
loadData()

cuda = torch.device('cuda')
model = AutoEncoderBridgeConv().cuda()
bridges = [Bridge().cuda() for b in range(model.hidden_size)]
loadModels()

optimizerBridges = [torch.optim.SGD(model.parameters(),lr=1e-2,momentum=1e-3) for ii in range(model.hidden_size)]
optimizerAutoencoder = torch.optim.Adam(model.parameters(),lr=1e-3,amsgrad=True)

LmuX = 0
LmuY = 0

imageLoss = 0
imageArray = []

while True:

    for batch in range(10):

        print('Batch num: ' + str(batch))

        torchimage = torch.Tensor(images[batch,:,1:785]).reshape([1000,1,28,28]).cuda()

        for epoch in range(10):
            AutoEncoderLossFunction(batch,epoch,torchimage,imageLoss)
            distributions.movetoCuda()
        for epoch in range(1000):
            BridgeLossFunction(batch,epoch,LmuX,LmuY)

        saveModels()

    imageArray.append(imageLoss)
