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
    def __init__(self, upsampling_factor=2):
        super(DownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        
    def forward(self, x, x_in, upsampling_factor=2):
        out = self.pool(x)
        out = x*torch.kron(x_in*1/out, torch.ones((upsampling_factor,upsampling_factor)).to('cuda'))
        return out

    
#network architecture
class ResNet(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, , upsampling_factor=2, noise=False, renorm=False):
        super(ResNet, self).__init__()
        # First layer
        if noise:
            self.conv_trans0 = nn.ConvTranspose2d(100, 1, kernel_size=(48,96), padding=0, stride=1)
            self.conv1 = nn.Sequential(nn.Conv2d(2, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(1, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #Residual Blocks
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        # Upsampling layers
        self.upsampling = nn.ModuleList()
        for k in range(np.rint(np.log2(upsampling_factor))):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2) )
        # Next layer after upper sampling
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        # Final output layer
        self.conv4 = nn.Conv2d(number_channels, 1, kernel_size=1, stride=1, padding=0)      
        #optional renomralization layer
        if downscale_constraints:
            self.downscale_constraint = DownscaleConstraints(upsampling_factor=upsampling_factor)
        
    if noise:
        def forward(self, x, z):
            out = self.conv_trans0(z)
            out = self.conv1(torch.cat((x,out), dim=1))
            out = self.res_blocks(out)
            out = self.conv2(out)
            out = self.upsampling(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if downscale_constraints:
                out = self.downscale_constraint(out)
            return out  
    else:
        def forward(self, x):
            out = self.conv1(x)
            out = self.res_blocks(out)
            out = self.conv2(out)
            out = self.upsampling(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if downscale_constraints:
                out = self.downscale_constraint(out)
            return out  
          
                    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True)) 
        self.conv9 = nn.Conv2d(128, 1, 1, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
    
