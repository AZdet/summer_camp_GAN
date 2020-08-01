import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm
from math import sqrt
import random
import pdb

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

def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()

def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out

class ConvDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True):
        super(ConvDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ConvUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConvUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_size, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

class AdaConvUp(nn.Module):
    def __init__(self, in_size, out_size, style_dim=512):
        super(AdaConvUp, self).__init__()
        self.upconv =  nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False)
        self.adain = AdaptiveInstanceNorm(out_size, style_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style):
        x = self.upconv(x)
        x = self.adain(x, style)
        x = self.relu(x)
        return x

class toRGB(nn.Module):
    def __init__(self, in_size, out_size):
        super(toRGB, self).__init__()
        self.conv =  nn.Conv2d(in_size, out_size, 3, 1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down1 = ConvDown(3, 64, normalize=False) # 64
        self.down2 = ConvDown(64, 128) # 32
        self.down3 = ConvDown(128, 256) # 16
        self.down4 = ConvDown(256, 512) # 8
        self.down5 = ConvDown(512, 512) # 4
        self.down6 = ConvDown(512, 512) # 2
        self.down7 = ConvDown(512, 128, normalize=False) # 1
        
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        return d7



class DisentangleDecoder(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super(DisentangleDecoder, self).__init__()
        self.up1 = ConvUp(128, 256) # 2
        self.up2 = AdaConvUp(256, 512) # 4 <- inject random variable
        self.up3 = AdaConvUp(512, 512) # 8 <- inject random variable
        self.up4 = AdaConvUp(512, 512) # 16 <- inject random variable
        self.up5 = AdaConvUp(512, 512) # 32 <- inject random variable
        self.up6 = AdaConvUp(512, 256) # 64 <- inject random variable
        self.up7 = AdaConvUp(256, 128) # 128 <- inject random variable

        self.toRGB_4 = toRGB(512, 3) # 4
        self.toRGB_8 = toRGB(512, 3) # 8        
        self.toRGB_16 = toRGB(512, 3) # 16
        self.toRGB_32 = toRGB(512, 3) # 32
        self.toRGB_64 = toRGB(256, 3) # 64
        self.toRGB_128 = toRGB(128, 3) # 128
        
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.style_layers = nn.Sequential(*layers)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z, style_codes):
        u1 = self.up1(z)
        style = self.style_layers(style_codes)
        u2 = self.up2(u1, style); rgb_4 = self.toRGB_4(u2) # 4
        u3 = self.up3(u2, style); rgb_8 = self.toRGB_8(u3) # 8
        u4 = self.up4(u3, style); rgb_16 = self.toRGB_16(u4) # 16
        u5 = self.up5(u4, style); rgb_32 = self.toRGB_32(u5) # 32
        u6 = self.up6(u5, style); rgb_64 = self.toRGB_64(u6) # 64
        u7 = self.up7(u6, style); rgb_128 = self.toRGB_128(u7) # 128   
        rgb_list = [rgb_4, rgb_8, rgb_16, rgb_32, rgb_64, rgb_128]
        return rgb_list

class Disc_4(nn.Module):
    def __init__(self):
        super(Disc_4, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 1, 2, 1, 0))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        z = self.conv3(x)
        return z

class Disc_8(nn.Module):
    def __init__(self):
        super(Disc_8, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 258, 3, 1, 1))
        self.conv4 = SpectralNorm(nn.Conv2d(258, 1, 2, 1, 0))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        z = self.conv4(x)
        return z

class Disc_16(nn.Module):
    def __init__(self):
        super(Disc_16, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 256, 4, 2, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1))
        self.conv4 = SpectralNorm(nn.Conv2d(512, 1, 2, 1, 0))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        z = self.conv4(x)
        return z

class Disc_32(nn.Module):
    def __init__(self):
        super(Disc_32, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 1, 2, 1, 0))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        z = self.conv5(x)
        return z

class Disc_64(nn.Module):
    def __init__(self):
        super(Disc_64, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1))
        self.conv6 = SpectralNorm(nn.Conv2d(1024, 1, 2, 1, 0))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        z = self.conv6(x)
        return z

class Disc_128(nn.Module):
    def __init__(self):
        super(Disc_128, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1))
        self.conv6 = SpectralNorm(nn.Conv2d(1024, 1024, 4, 1, 0))
        self.conv7 = SpectralNorm(nn.Conv2d(1024, 1, 1, 1, 0))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        z = self.conv7(x)
        return z

def denorm(tensor):
	return ((tensor+1.0)/2.0)*255.0

def norm(image):
	return (image/255.0-0.5)*2.0

if __name__ == '__main__':
        
    num_sample = 2 # number of faces you want to sample
    
    rgb = torch.ones((1,3,128,128))
    mask = torch.ones((1,3,128,128))
    
    # process inptus
    mask[mask>0] = 1
    cond_input = rgb*mask
    rgb, cond_input = norm(rgb), norm(cond_input)
    
    # load models
    encoder = torch.load('encoder_29.pt', map_location=lambda storage, loc: storage)
    decoder = torch.load('decoder_29.pt', map_location=lambda storage, loc: storage)
    
    # Sample Style Code 
    batch = rgb.shape[0]
    style_codes = torch.FloatTensor(batch, num_sample, 512).uniform_().view(-1,512)
    
    # Encode occluded face image 
    code = encoder(cond_input)
    	
    # Decode face image
    rgb_hat = decoder(code, style_codes)[-1]
    
    pdb.set_trace()










