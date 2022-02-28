import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#building blocks for networks
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class DownscaleConstraints(nn.Module):
    def __init__(self, upsampling_factor):
        super(DownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        sum_y = self.pool(y)
        out = y*torch.kron(lr*1/sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    
    
class SoftmaxConstraints(nn.Module):
    def __init__(self, upsampling_factor):
        super(SoftmaxConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        y = torch.exp(y)
        sum_y = self.pool(y)
        out = y*torch.kron(lr*1/sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out

    
#network architecture
class ResNet1(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, downscale_constraints=False, softmax_constraints=False, dim=2):
        super(ResNet1, self).__init__()
        # First layer
        if noise:
            self.conv_trans0 = nn.ConvTranspose2d(100, 1, kernel_size=(32,32), padding=0, stride=1)
            self.conv1 = nn.Sequential(nn.Conv2d(2, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #Residual Blocks
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        # Upsampling layers
        self.upsampling = nn.ModuleList()
        for k in range(int(np.rint(np.log2(upsampling_factor)))):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2) )
        # Next layer after upper sampling
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        # Final output layer
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)      
        #optional renomralization layer
        if downscale_constraints:
            if softmax_constraints:
                self.downscale_constraint = SoftmaxConstraints(upsampling_factor=upsampling_factor)
            else:
                self.downscale_constraint = DownscaleConstraints(upsampling_factor=upsampling_factor)
            
        self.noise = noise
        self.downscale_constraints = downscale_constraints
    def forward(self, x, z=None): 
        if self.noise:
            out = self.conv_trans0(z)
            out = self.conv1(torch.cat((x,out), dim=1))
            for layer in self.res_blocks:
                out = layer(out)
            out = self.conv2(out)
            for layer in self.upsampling:
                out = layer(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if self.downscale_constraints:
                out = self.downscale_constraint(out, x)
            return out  
        else:
            
            out = self.conv1(x)
            for layer in self.res_blocks:
                out = layer(out)
            out = self.conv2(out)
            for layer in self.upsampling:
                out = layer(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if self.downscale_constraints:
                out = self.downscale_constraint(out, x)
            return out  
        
        
class ResNet2(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, downscale_constraints=False, softmax_constraints=False):
        super(ResNet2, self).__init__()
        # First layer
        if noise:
            self.conv_trans0 = nn.ConvTranspose2d(100, 1, kernel_size=(32,32), padding=0, stride=1)
            self.conv1 = nn.Sequential(nn.Conv2d(2, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(2, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #Residual Blocks
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        # Upsampling layers
        self.upsampling = nn.ModuleList()
        for k in range(int(np.rint(np.log2(upsampling_factor)))):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2) )
        # Next layer after upper sampling
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        # Final output layer
        self.conv4 = nn.Conv2d(number_channels, 2, kernel_size=1, stride=1, padding=0)      
        #optional renomralization layer
        if downscale_constraints:
            if softmax_constraints:
                self.downscale_constraint = SoftmaxConstraints(upsampling_factor=upsampling_factor)
            else:
                self.downscale_constraint = DownscaleConstraints(upsampling_factor=upsampling_factor)
            
        self.noise = noise
        self.downscale_constraints = downscale_constraints
    def forward(self, x, z=None): 
        if self.noise:
            out = self.conv_trans0(z)
            out = self.conv1(torch.cat((x,out), dim=1))
            for layer in self.res_blocks:
                out = layer(out)
            out = self.conv2(out)
            for layer in self.upsampling:
                out = layer(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if self.downscale_constraints:
                out = self.downscale_constraint(out, x)
            return out  
        else:
            
            out = self.conv1(x)
            for layer in self.upsampling:
                out = layer(out)
            out = self.conv2(out)    
            for layer in self.res_blocks:
                out = layer(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if self.downscale_constraints:
                out = self.downscale_constraint(out, x)
            return out  
          
                    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True)) 
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True)) 
        self.conv9 = nn.Conv2d(128, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
    
