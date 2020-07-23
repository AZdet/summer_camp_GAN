import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pdb
from spectral_normalization import SpectralNorm

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_size, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# G(z)
class Decoder(nn.Module):
    # initializers
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = UNetUp(100+16, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)
        
        final = [   nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
                    nn.Tanh() ]
        self.final = nn.Sequential(*final) 

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z, feats):
        [d1, d2, d3, d4, d5, d6] = feats
        u1 = self.up1(z, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)

class Encoder(nn.Module):
    def __init__(self, d=16):
        super(Encoder, self).__init__()
        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)
        
        self.fc = nn.Linear(512,100)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d7 = d7.view(d7.shape[0], -1) 
        return self.fc(d7), [d1,d2,d3,d4,d5,d6]

    def sample(self, mu, logvar, noise):
        std = torch.exp(0.5*logvar)
        return noise.mul(std).add_(mu)

    def kl_div(self, mu, logvar):
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1))
        self.conv6 = SpectralNorm(nn.Conv2d(1024, 1024, 4, 1, 0))
        self.conv7 = SpectralNorm(nn.Conv2d(1024, 1, 1, 1, 0))

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # print('input: ', input.shape)
        x = F.leaky_relu(self.conv1(input), 0.2)
        # print('1: ', x.shape)
        x = F.leaky_relu(self.conv2(x), 0.2)
        # print('2: ', x.shape)
        x = F.leaky_relu(self.conv3(x), 0.2)
        # print('3: ', x.shape)
        x = F.leaky_relu(self.conv4(x), 0.2)
        # print('4: ', x.shape)
        x = F.leaky_relu(self.conv5(x), 0.2)
        # print('5: ', x.shape)
        x = F.leaky_relu(self.conv6(x), 0.2)
        # print('6: ', x.shape)
        z = self.conv7(x)
        # print('7: ', x.shape)
        # pdb.set_trace()
        return z




if __name__ == '__main__':
    x = torch.ones((1,3,128,128))
    vencoder, decoder, discriminator = VEncoder(), Decoder(), discriminator()
    mu, logvar, feats = vencoder(x)

    dyna_noise = torch.randn(mu.size())
    dyna_code = vencoder.sample(mu, logvar, dyna_noise).unsqueeze(2).unsqueeze(3)
    high_hat = decoder(dyna_code,feats)
    z = discriminator(high_hat)
    pdb.set_trace()




