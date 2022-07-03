import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

motif_basis = Variable(torch.load('./data/basis_4.pt'),requires_grad=True).to('cuda')

# To use the following code, you need to copy them to your "models.py" file.
# Author: Qidong Yang
# Last Edit Date: 06-24-2022


'''

Learnable Basis Module

'''


class BasisGenerator(nn.Module):
    def __init__(self, shape=(64, 8, 8), positivity=True, device='cuda'):
        super(BasisGenerator, self).__init__()

        self.shape = shape
        self.positivity = positivity
        self.size = shape[0] * shape[1] * shape[2]
        self.device = device

        self.basis_generator = nn.Linear(1, self.size, bias=False)

        # Generate learnable basis with sum one on shape[0] dimension
        # shape: the basis output shape without batch dimension [tuple] (reduce shape[0] to encourage sparsity)
        # positivity: whether basis elementwise positive [boolean]
        # device: the place to put the basis

    def forward(self):

        input = torch.ones(1, 1).to(self.device)
        basis = self.basis_generator(input)
        # (1, size)
        basis = basis.reshape((1, self.shape[0], self.shape[1], self.shape[2]))[0]
        # (shape[0], shape[1], shape[2])

        if self.positivity:
            basis = torch.exp(basis)

        # Normalization
        sums = torch.sum(basis, dim=(1, 2))
        basis = basis / sums.reshape((-1, 1, 1))

        return basis


'''

MotifNet with learnable basis example
Note: the following code serves as an example to show how to use the above learnable basis module,
which means the following code has not been tested yet!!

'''


class MotifNetLearnBasis(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, constraints='none', dim=1):
        super(MotifNetLearnBasis, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        # Upsampling layers
        self.upsampling = nn.ModuleList()
        for k in range(int(np.rint(np.log2(upsampling_factor)))):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2))

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, upsampling_factor**2, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))

        # Constraint layers
        self.softmax = nn.Softmax(dim=1)
        self.mult_in = MultIn()
        self.basis_generator = BasisGenerator(shape=(upsampling_factor**2, upsampling_factor, upsampling_factor))

    def forward(self, x):

        out = self.conv1(x[:, 0, ...])
        out = self.conv2(out)

        for layer in self.res_blocks:
            out = layer(out)

        out = self.conv3(out)

        # Softmax constraining
        out = self.softmax(out)
        # (n_batch, 16, 32, 32)
        out = self.mult_in(out, x[:, 0, ...])
        # (n_batch, 16, 32, 32)
        # Generate basis
        basis = self.basis_generator()
        # (16, 4, 4)

        # Kron operation
        output = Variable(torch.zeros((out.shape[0], out.shape[1], out.shape[2] * basis.shape[-2], out.shape[3] * basis.shape[-1])), requires_grad=True).to('cuda')
        for ii in range(basis.shape[0]):
            output[:, ii, :, :] = torch.kron(out[:, ii, :, :], basis[ii, :, :])
        # (n_batch, 16, 32*4, 32*4)

        # Add channels
        output = torch.sum(output, dim=1)
        output = output.unsqueeze(1)
        output = output.unsqueeze(1)

        return output


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
    
    
class ResidualUpsampling(nn.Module):
    def __init__(self, in_channels=8, out_channels=64, stride=1, downsample=None):
        super(ResidualUpsampling, self).__init__()
        self.res1 = ResidualBlock(in_channels, in_channels, stride=1)
        self.res2 = ResidualBlock(in_channels, in_channels, stride=1)
        self.res3 = ResidualBlock(in_channels, out_channels, stride=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.up(x)
        return x
    

class MultDownscaleConstraints(nn.Module):
    def __init__(self, upsampling_factor):
        super(MultDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        sum_y = self.pool(y)
        out = y*torch.kron(lr*1/sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    
    
class AddDownscaleConstraints(nn.Module):
    def __init__(self, upsampling_factor):
        super(AddDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        sum_y = self.pool(y)
        out =y+ torch.kron(lr-sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    
class EnforcementOperator(nn.Module):
    def __init__(self, upsampling_factor):
        super(EnforcementOperator, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        sum_y = self.pool(y)
        diff_P_x = torch.kron(lr-sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        sigma = torch.sign(-diff_P_x)
        out =y+ diff_P_x*(sigma+y)/(sigma+torch.kron(sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda')))
        return out 
    
class SoftmaxConstraints(nn.Module):
    def __init__(self, upsampling_factor, exp_factor=1):
        super(SoftmaxConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
        self.exp_factor = exp_factor
    def forward(self, y, lr):
        y = torch.exp(y/self.exp_factor)
        sum_y = self.pool(y)
        out = y*torch.kron(lr*1/sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    

class Motifs(nn.Module):
    def __init__(self):
        super(Motifs, self).__init__()
    def forward(self, y):
        out = Variable(torch.zeros((y.shape[0],y.shape[1],y.shape[2]*4,y.shape[3]*4)), requires_grad=True).to('cuda')
        for i in range(16):
            out[:,i,...] = torch.kron(y[:,i,...], motif_basis[i,...])
        return out
        
class MultIn(nn.Module):
    def __init__(self):
        super(MultIn, self).__init__()
    def forward(self, y, lr):
        return 16*y*lr
    
class AddChannels(nn.Module):
    def __init__(self):
        super(AddChannels, self).__init__()
    def forward(self, y):
        return torch.sum(y, dim=1).unsqueeze(1)
        
    
class MRSoftmaxConstraints(nn.Module):
    def __init__(self, upsampling_factor, exp_factor=1):
        super(MRSoftmaxConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=int(upsampling_factor/2))
        self.upsampling_factor = int(upsampling_factor/2)
        self.exp_factor = exp_factor
    def forward(self, y, mr):
        y = torch.exp(self.exp_factor*y)
        sum_y = self.pool(y)
        out = y*torch.kron(mr*1/sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    
#network architecture
class ResNet(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, downscale_constraints=False, softmax_constraints=False, dim=2):
        super(ResNet, self).__init__()
        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(1, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.res_up1 = ResidualUpsampling(number_channels, number_channels)
        self.res_up2 = ResidualUpsampling(number_channels, number_channels)
        self.upsampling = nn.ModuleList()
        for k in range(2):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2) )
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        self.softmax_constraints = softmax_constraints
        if softmax_constraints:
            self.downscale_constraint = SoftmaxConstraints(upsampling_factor=upsampling_factor)
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, 1, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))    
    def forward(self, x):  
        out = self.conv1(x[:,0,...])
        #out = self.res_up1(out)
        #out = self.res_up2(out)
        for layer in self.upsampling:
                out = layer(out)
        #for layer in self.res_blocks:
        #    out = layer(out)
        out = self.conv2(out)
        if self.softmax_constraints:
            out = self.downscale_constraint(out, x)
        out = out.unsqueeze(1)
        return out  
    
class CNNUp(nn.Module):
    def __init__(self):
        super(CNNUp,self).__init__()
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=True))
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        
    def forward(self, x):
        out = self.up1(x[:,0,...])
        out = self.conv1(out)
        out = self.up2(out)
        out = self.conv2(out)
        return out.unsqueeze(1)
        


    
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
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, constraints='none', dim=1):
        super(ResNet2, self).__init__()
        # First layer
        if noise:
            self.conv_trans0 = nn.ConvTranspose2d(100, 1, kernel_size=(32,32), padding=0, stride=1)
            self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #Residual Blocks
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        # Upsampling layers
        self.upsampling = nn.ModuleList()
        for k in range(int(np.rint(np.log2(upsampling_factor)))):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2) )
        # Next layer after upper sampling
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        # Final output layer
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)      
        #optional renomralization layer
        self.is_constraints = False
        if constraints == 'softmax':
            self.constraints = SoftmaxConstraints(upsampling_factor=upsampling_factor)
            self.is_constraints = True
        elif constraints == 'enforce_op':
            self.constraints = EnforcementOperator(upsampling_factor=upsampling_factor)
            self.is_constraints = True
        elif constraints == 'add':
            self.constraints = AddDownscaleConstraints(upsampling_factor=upsampling_factor)
            self.is_constraints = True
        elif constraints == 'mult':
            self.constraints = MultDownscaleConstraints(upsampling_factor=upsampling_factor)
            self.is_constraints = True
            
            
        self.noise = noise
        
    def forward(self, x, mr=None, z=None): 
        if self.noise:
            out = self.conv_trans0(z)
            out = self.conv1(torch.cat(( x[:,0,...],out), dim=1))
            for layer in self.res_blocks:
                out = layer(out)
            out = self.conv2(out)
            for layer in self.upsampling:
                out = layer(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if self.is_constraints:
                out = self.constraints(out,  x[:,0,...])
            return out  
        else:
            #print(x.shape)
            out = self.conv1(x[:,0,...])
            for layer in self.upsampling:
                out = layer(out)
            out = self.conv2(out)    
            for layer in self.res_blocks:
                out = layer(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if self.is_constraints:
                out = self.constraints(out, x[:,0,...])
            #out[:,0,:,:] *= 16
            out = out.unsqueeze(1)
            return out  
        
class MotifNet(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, constraints='none', dim=1):
        super(MotifNet, self).__init__()
        # First layer
        
        self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #Residual Blocks
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        # Upsampling layers
        self.upsampling = nn.ModuleList()
        for k in range(int(np.rint(np.log2(upsampling_factor)))):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2) )
        # Next layer after upper sampling
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        self.mult_in = MultIn()
        self.motifs = Motifs()
        self.add_channels = AddChannels()
        self.sum_channels = AddChannels()
        self.constraints = constraints
        if constraints == 'softmax_last':
            self.constraints_layer = SoftmaxConstraints(upsampling_factor=upsampling_factor)
        elif constraints == 'sum_last':
            self.constraints_layer = MultDownscaleConstraints(upsampling_factor=upsampling_factor)
        
        
    def forward(self, x): 
        out = self.conv1(x[:,0,...])
        out = self.conv2(out)  
        for layer in self.res_blocks:
            out = layer(out)
        out = self.conv3(out)
        if self.constraints == 'softmax_first':
            out = self.softmax(out)
            out = self.mult_in(out, x[:,0,...])
        elif self.constraints == 'sum_first':
            sum_c = self.sum_channels(out)
            out = out/sum_c
            out = self.mult_in(out, x[:,0,...])
        out = self.motifs(out)
        out = self.add_channels(out)
        if self.constraints == 'softmax_last' or self.constraints == 'sum_last':
            out = self.constraints_layer(out, x[:,0,...])
        out = out.unsqueeze(1)
        return out  
        
        

        
class ResNet2Up(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, constraints='none', dim=1, output_mr=False):
        super(ResNet2Up, self).__init__()
        #PART I
        self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #upsampling
        self.up1 = nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)      
        
        self.is_constraints = False
        if constraints == 'softmax':
            self.constraints = SoftmaxConstraints(upsampling_factor=2)
            self.is_constraints = True
        elif constraints == 'enforce_op':
            self.constraints = EnforcementOperator(upsampling_factor=2)
            self.is_constraints = True
            
        #PART II
        self.conv21 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.res_blocks2 = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks2.append(ResidualBlock(number_channels, number_channels))
        self.conv22 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #upsampling
        self.up2 = nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2)
        self.conv23 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv24 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)
        if constraints == 'softmax':
            self.constraints2 = SoftmaxConstraints(upsampling_factor=2)
        elif constraints == 'enforce_op':
            self.constraints2 = EnforcementOperator(upsampling_factor=2)
 
        self.output_mr = output_mr             
    def forward(self, x, mr_in=None):
        #part 1
        out = self.conv1(x[:,0,...])
        out = self.up1(out)
        out = self.conv2(out)    
        for layer in self.res_blocks:
            out = layer(out)
        out = self.conv3(out)
        mr = self.conv4(out)
        if self.is_constraints:
            mr = self.constraints(mr, x[:,0,...])
        #part 2
        out = self.conv21(mr)
        out = self.up2(out)
        out = self.conv22(out)    
        for layer in self.res_blocks2:
            out = layer(out)
        out = self.conv23(out)
        out = self.conv24(out)
        if self.is_constraints:
            out = self.constraints2(out, mr)
        if self.output_mr:
            return out.unsqueeze(1), mr.unsqueeze(1)
        else:
            return out.unsqueeze(1)
        
        
class ResNet3Up(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, constraints='none', dim=1, output_mr=False):
        super(ResNet3Up, self).__init__()
        #PART I
        self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #upsampling
        self.up1 = nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)      
        
        self.is_constraints = False
        if constraints == 'softmax':
            self.constraints = SoftmaxConstraints(upsampling_factor=2)
            self.is_constraints = True
        elif constraints == 'enforce_op':
            self.constraints = EnforcementOperator(upsampling_factor=2)
            self.is_constraints = True
            
       
            
            
        #PART II
        self.conv21 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.res_blocks2 = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks2.append(ResidualBlock(number_channels, number_channels))
        self.conv22 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #upsampling
        self.up2 = nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2)
        self.conv23 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv24 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)
        if constraints == 'softmax':
            self.constraints2 = SoftmaxConstraints(upsampling_factor=2)
        elif constraints == 'enforce_op':
            self.constraints2 = EnforcementOperator(upsampling_factor=2)
            
            
         #PART III
        self.conv31 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.res_blocks3 = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks3.append(ResidualBlock(number_channels, number_channels))
        self.conv32 = nn.Sequential(
            nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #upsampling
        self.up3 = nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2)
        self.conv33 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        self.conv34 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)
        if constraints == 'softmax':
            self.constraints3 = SoftmaxConstraints(upsampling_factor=2)
        elif constraints == 'enforce_op':
            self.constraints3 = EnforcementOperator(upsampling_factor=2)
 
        self.output_mr = output_mr             
    def forward(self, x, mr_in=None):
        #part 1
        
        out = self.conv1(x[:,0,...])
        out = self.up1(out)
        out = self.conv2(out)    
        for layer in self.res_blocks:
            out = layer(out)
        out = self.conv3(out)
        mr1 = self.conv4(out)
        if self.is_constraints:
            mr1 = self.constraints(mr1, x[:,0,...])
        #part 2
        out = self.conv21(mr1)
        out = self.up2(out)
        out = self.conv22(out)    
        for layer in self.res_blocks2:
            out = layer(out)
        out = self.conv23(out)
        mr2 = self.conv24(out)
        if self.is_constraints:
            mr2 = self.constraints2(mr2, mr1)
            
        #part 3
        out = self.conv31(mr2)
        out = self.up3(out)
        out = self.conv32(out)    
        for layer in self.res_blocks3:
            out = layer(out)
        out = self.conv33(out)
        out = self.conv34(out)
        if self.is_constraints:
            out = self.constraints3(out, mr2)
        if self.output_mr:
            return out.unsqueeze(1), mr.unsqueeze(1)
        else:
            return out.unsqueeze(1)
        
        
class MRResNet(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, downscale_constraints=False, softmax_constraints=False, dim=1, output_mr=False):
        super(MRResNet, self).__init__()
        # First layer
        if noise:
            self.conv_trans0 = nn.ConvTranspose2d(100, 1, kernel_size=(32,32), padding=0, stride=1)
            self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(dim, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        #Residual Blocks
        self.res_blocks1 = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks1.append(ResidualBlock(number_channels, number_channels))
        self.res_blocks2 = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks2.append(ResidualBlock(number_channels, number_channels))
        self.res_blocks3 = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks3.append(ResidualBlock(number_channels, number_channels))
        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(number_channels,number_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        # Upsampling layers
        self.upsampling1 = nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2) 
        self.upsampling2 = nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2) 
        
        # Next layer after upper sampling
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        # Final output layer
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0) 
        self.conv5 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0) 
        self.conv6 = nn.Conv2d( dim,number_channels, kernel_size=1, stride=1, padding=0) 
        #optional renomralization layer
        self.downscale_constraint1 = SoftmaxConstraints(upsampling_factor=int(upsampling_factor/2))
        self.downscale_constraint2 = SoftmaxConstraints(upsampling_factor=int(upsampling_factor/2))
        
            
        self.noise = noise
        self.output_mr = output_mr
        
    def forward(self, x, mr=None): 
        '''
        out = self.conv1(x[:,0,...])
        out = self.upsampling1(out)
        for layer in self.res_blocks1:
            out = layer(out) 
        out = self.conv2(out)
        #mr = self.downscale_constraint1(out, x[:,0,...])
        #if not mr:
           # mr = torch.clone(out)
        
        out = self.conv3(out)
        
        #for layer in self.res_blocks2:
        #    out = layer(out)
        out = self.upsampling2(out)
        for layer in self.res_blocks3:
            out = layer(out)
        out = self.conv4(out)
        out = self.downscale_constraint2(out,  x[:,0,...])
        if self.output_mr:
            return out.unsqueeze(1), mr.unsqueeze(1)
        else:
            return out.unsqueeze(1)'''
        out = self.conv1(x[:,0,...])
        out = self.upsampling1(out)
        out = self.conv2(out) 
        for layer in self.res_blocks1:
            out = layer(out)
        mr = self.conv5(out)
        #mr = self.downscale_constraint1(out, x[:,0,...])
        out = self.conv6(mr)
        for layer in self.res_blocks2:
            out = layer(out)
        out = self.upsampling2(out)
        out = self.conv3(out)   
        for layer in self.res_blocks3:
            out = layer(out)
        
        out = self.conv4(out)
        
        #out = self.downscale_constraint2(out, mr)
        #out[:,0,:,:] *= 16
        out = out.unsqueeze(1)
        if self.output_mr:
            return out, mr.unsqueeze(1)
        else:
            return out  
        
###successive constrainning
        
###ESRGAN 
        
class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.
    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out
    
    
class ESRGANGenerator(nn.Module):
    def __init__(self) -> None:
        super(ESRGANGenerator, self).__init__()
        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(23):
            trunk.append(ResidualResidualDenseBlock(64, 32))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        self.upsampling1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.upsampling2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x[:,0,...])
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out.unsqueeze(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
    
    
###SRGAN generator
def swish(x):
    return x * F.sigmoid(x)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))

class SRGANGenerator(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(SRGANGenerator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(int(self.upsample_factor/2)):
            self.add_module('upsample' + str(i+1), upsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 1, 9, stride=1, padding=4)

    def forward(self, x):
        x = swish(self.conv1(x[:,0,...]))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = self.bn2(self.conv2(y)) + x

        for i in range(int(self.upsample_factor/2)):
            x = self.__getattr__('upsample' + str(i+1))(x)
        x = self.conv3(x)
        return x.unsqueeze(1)

        

          
class ResNetNoise(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, downscale_constraints=False, softmax_constraints=False, dim=1):
        super(ResNetNoise, self).__init__()
        # First layer
        
        self.linear = nn.Linear(100, 32*32)
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
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)      
        #optional renomralization layer
        if downscale_constraints:
            if softmax_constraints:
                self.downscale_constraint = SoftmaxConstraints(upsampling_factor=upsampling_factor)
            else:
                self.downscale_constraint = DownscaleConstraints(upsampling_factor=upsampling_factor)
            
        self.noise = noise
        self.downscale_constraints = downscale_constraints
    def forward(self, x, z): 
        out = self.linear(z)
        x = x[:,0,...]
        out = torch.reshape(out, (x.shape))
        out = self.conv1(torch.cat((x,out), dim=1))
        
        
        for layer in self.upsampling:
            out = layer(out)
        out = self.conv2(out)    
        for layer in self.res_blocks:
            out = layer(out)
        out = self.conv3(out)
        out = self.conv4(out)
        if self.downscale_constraints:
            out = self.downscale_constraint(out, x)
        out = out.unsqueeze(1)
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
        x = x[:,0,...]
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)
    
####time-series model
###work in progress
### data shape (batch_size, time_steps=8, channels=1, H, W)


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        shape = x.shape
        if len(shape)==5:
            x_reshape = x.reshape(shape[0]*shape[1], shape[2],shape[3], shape[4])  # (samples * timesteps, input_size)
            y = self.module(x_reshape)
            y = y.reshape(shape[0],shape[1],y.shape[1],y.shape[2],y.shape[3])
        elif len(shape)==4:
            x_reshape = x.reshape(shape[0]*shape[1], shape[2],shape[3])
            y = self.module(x_reshape)
            y = y.reshape(shape[0],shape[1],y.shape[1],y.shape[2])
        elif len(shape)==3:
            x_reshape = x.reshape(shape[0]*shape[1], shape[2])
            y = self.module(x_reshape)
            y = y.reshape(shape[0],shape[1],y.shape[1])
        
        # We have to reshape Y
        
        return y
    
class SoftmaxConstraintsTime(nn.Module):
    def __init__(self, upsampling_factor, exp_factor=1):
        super(SoftmaxConstraintsTime, self).__init__()
        self.pool = TimeDistributed(torch.nn.AvgPool2d(kernel_size=upsampling_factor))
        self.upsampling_factor = upsampling_factor
        self.exp_factor = exp_factor
    def forward(self, y, lr):
        y = torch.exp(self.exp_factor*y)
        sum_y = self.pool(y)
        out = y*torch.kron(lr*1/sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out


class GenGate(nn.Module):
    def __init__(self, activation='sigmoid', number_of_inchannels=64, number_of_outchannels=64):
        super(GenGate, self).__init__()
        self.refl = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(number_of_inchannels, number_of_outchannels, kernel_size=(3,3))
        if activation is not None:
            self.act = nn.Sigmoid()
        else:
            self.act = None
               
    def forward(self, x):
        x = self.refl(x)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x
            
            
class GenGateGRU(nn.Module):
    def __init__(self, return_sequences=True, time_steps=8):
        super(GenGateGRU, self).__init__()
        self.update_gate = GenGate('sigmoid', 128,64)
        self.reset_gate = GenGate('sigmoid', 128,64)
        self.output_gate = GenGate(None, 128,64)
        self.return_sequences = return_sequences
        self.time_steps = time_steps

    def forward(self, inputs):
        (xt,h) = inputs
        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            xh = torch.cat((x,h), dim=1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(torch.cat((x,r*h), dim=1))
            h = z*h + (1-z)*torch.tanh(o)
            if self.return_sequences:
                h_all.append(h)
        return torch.stack(h_all,dim=1) if self.return_sequences else h

                

class ResidualBlockRNN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='leaky_relu'):
        super(ResidualBlockRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='reflect')
        if activation == 'relu':
            self.relu1 = nn.ReLU( inplace=False )
        elif activation == 'leaky_relu':
            self.relu1 = nn.LeakyReLU( inplace=False )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='reflect')
        if activation == 'relu':
            self.relu2 = nn.ReLU( inplace=False )
        elif activation == 'leaky_relu':
            self.relu2 = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu1(x)
        out = TimeDistributed(self.conv1)(out)
        out = self.relu2(out)
        out = TimeDistributed(self.conv2)(out)
        out += residual
        return out
        
class ResidualBlockRNNSpectral(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='leaky_relu'):
        super(ResidualBlockRNNSpectral, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool = TimeDistributed(nn.AvgPool2d(kernel_size=(stride,stride)))
        self.conv0 = TimeDistributed(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='reflect'))
        if activation == 'relu':
            self.relu1 = nn.ReLU(inplace=False)
        elif activation == 'leaky_relu':
            self.relu1 = nn.LeakyReLU(inplace=False)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'))
        if activation == 'relu':
            self.relu2 = nn.ReLU( inplace=False)
        elif activation == 'leaky_relu':
            self.relu2 = nn.LeakyReLU( inplace=False )
        
    def forward(self, x):
        residual = x
        if self.stride > 1:
            residual = self.pool(residual)
        if not self.in_channels==self.out_channels:
            residual = self.conv0(residual)
        out = self.relu1(x)
        out = TimeDistributed(self.conv1)(out)
        out = self.relu2(out)
        out = TimeDistributed(self.conv2)(out)
        out += residual
        return out
    
    
        
class ResidualBlockN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation='leaky_relu'):
        super(ResidualBlockN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='reflect')
        if activation == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='reflect')
        if activation == 'relu':
            self.relu2 = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.relu2 = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out += residual
        return out
        

    
class InitialState(nn.Module):
    def __init__(self, number_channels=64,  number_residual_blocks=3):
        super(InitialState, self).__init__()    
        self.conv = nn.Conv2d(1, number_channels-8,kernel_size=(3,3), padding=1)
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlockN(number_channels, number_channels, stride=1, activation='relu'))
        
    def forward(self, x, noise):
        out = self.conv(x)
        out = torch.cat((out, noise), dim=1)
        for layer in self.res_blocks:
            out = layer(out)
        return out
    
class InitialStateDet(nn.Module):
    def __init__(self, number_channels=64,  number_residual_blocks=3):
        super(InitialStateDet, self).__init__()
        self.conv = nn.Conv2d(1, number_channels,kernel_size=(3,3), padding=1)
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlockN(number_channels, number_channels, stride=1, activation='relu'))
        
    def forward(self, x):
        #out = self.reflpadd(x)
        out = self.conv(x)
        for layer in self.res_blocks:
            out = layer(out)
        return out
        
        
        
class DiscriminatorRNN(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=3):
        super(DiscriminatorRNN, self).__init__()
        channels = [1, 32, 64]
        self.downsampling_hr = nn.ModuleList()
        self.downsampling_lr = nn.ModuleList()
        for i in range(2):        
            self.downsampling_hr.append(ResidualBlockRNNSpectral(channels[i], channels[i+1], stride=2, activation='leaky_relu'))
            self.downsampling_lr.append(ResidualBlockRNNSpectral(channels[i], channels[i+1], stride=1, activation='leaky_relu'))
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            if k == 0:
                self.res_blocks.append(ResidualBlockRNNSpectral(number_channels*2, number_channels, stride=1, activation='leaky_relu'))
            else:
                self.res_blocks.append(ResidualBlockRNNSpectral(number_channels, number_channels, stride=1, activation='leaky_relu'))
        self.res_blocks_hr = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks_hr.append(ResidualBlockRNNSpectral(number_channels, number_channels, stride=1, activation='leaky_relu'))
        self.conv_gru_joint = DiscGateGRU(return_sequences=True, time_steps=8)  
        self.conv_gru_hr = DiscGateGRU(return_sequences=True, time_steps=8) 
        #self.avg_joint = TimeDistributed(nn.AvgPool2d(kernel_size=(32,32)))
        #self.avg_hr = TimeDistributed(nn.AvgPool2d(kernel_size=(32,32)))
        self.sn_dense = TimeDistributed(nn.utils.spectral_norm(nn.Linear(128,64)))
        self.leakyrelu = nn.LeakyReLU( inplace=False )
        self.sn_dense_one = TimeDistributed(nn.utils.spectral_norm(nn.Linear(64,1) ))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, hr, lr):
        for layer in self.downsampling_hr:
            hr = layer(hr)
        for layer in self.downsampling_lr:
            lr = layer(lr)
        joint = torch.cat((lr, hr), dim=2)
        for layer in self.res_blocks:
            joint = layer(joint)
        for layer in self.res_blocks_hr:
            hr = layer(hr)
        joint = self.conv_gru_joint([joint, torch.zeros_like(joint[:,0,...])])
        hr = self.conv_gru_hr([hr, torch.zeros_like(hr[:,0,...])])
        joint = torch.mean(joint, dim=(3,4))
        hr = torch.mean(hr, dim=(3,4))
        out = torch.cat((joint, hr),dim=2)
        out = self.sn_dense(out)
        #out = out.reshape(out.shape[0], out.shape[1], out
        out = self.leakyrelu(out)
        out = self.sn_dense_one(out)
        out = self.sigmoid(out)
        return out           

class DiscGate(nn.Module):
    def __init__(self, activation='sigmoid', number_of_inchannels=64, number_of_outchannels=64):
        super(DiscGate, self).__init__()
        self.refl = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(number_of_inchannels, number_of_outchannels, kernel_size=(3,3)))
        if activation is not None:
            self.act = nn.Sigmoid()
        else:
            self.act = None
               
    def forward(self, x):
        x = self.refl(x)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x
            
class DiscGateGRU(nn.Module):
    def __init__(self, return_sequences=True, time_steps=8):
        super(DiscGateGRU, self).__init__()
        self.update_gate = DiscGate('sigmoid', 128,64)
        self.reset_gate = DiscGate('sigmoid', 128,64)
        self.output_gate = DiscGate(None, 128,64)
        self.return_sequences = return_sequences
        self.time_steps = time_steps

    def forward(self, inputs):
        (xt,h) = inputs
        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            xh = torch.cat((x,h), dim=1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(torch.cat((x,r*h), dim=1))
            h = z*h + (1-z)*torch.tanh(o)
            if self.return_sequences:
                h_all.append(h)
        return torch.stack(h_all,dim=1) if self.return_sequences else h


    
    
class ConvGRUGenerator(nn.Module):    
    def __init__(self, number_channels=64, number_residual_blocks=3, upsampling_factor=2, time_steps=3):
        super(ConvGRUGenerator, self).__init__()  
        self.initialize = InitialState()
        self.reflpadd1 = TimeDistributed(nn.ReflectionPad2d(1))
        self.conv1 = TimeDistributed(nn.Conv2d(1, number_channels-8, kernel_size=(3,3), padding=1))
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlockRNN(number_channels, number_channels, stride=1, activation='relu'))
        self.convgru = GenGateGRU(return_sequences=True, time_steps=time_steps)  
        self.upsampling = nn.ModuleList()
        for i in range(3):
            if i > 0:
                self.upsampling.append(TimeDistributed(nn.UpsamplingBilinear2d(scale_factor=2)))
            self.upsampling.append(ResidualBlockRNN(number_channels, number_channels, stride=1, activation='leaky_relu'))          
        self.reflpadd2 = TimeDistributed(nn.ReflectionPad2d(1))
        self.conv2 = TimeDistributed(nn.Conv2d(number_channels, 1, kernel_size=(3,3), padding=1))
        self.constraint = SoftmaxConstraintsTime(upsampling_factor=4)
    
    def forward(self, low_res, noise, initial_noise):   
        initial_state = self.initialize(low_res[:,0,...], initial_noise)
        #xt = self.reflpadd1(low_res)
        
        xt = self.conv1(low_res)
        xt = torch.cat((xt, noise), dim=2)
        for layer in self.res_blocks:
            xt = layer(xt)
        x = self.convgru([xt, initial_state])
        h = x[:,-1,...]
        for layer in self.upsampling:
            x = layer(x)           
        #x = self.reflpadd2(x)
        img_out = self.conv2(x)   
        img_out = self.constraint(img_out, low_res)
        return img_out
    
class ConvGRUGeneratorDet(nn.Module):    
    def __init__(self, number_channels=64, number_residual_blocks=3, upsampling_factor=2, time_steps=3):
        super(ConvGRUGeneratorDet, self).__init__()  
        self.initialize = InitialStateDet()
        self.conv1 = TimeDistributed(nn.Conv2d(1, number_channels, kernel_size=(3,3), padding=1))
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlockRNN(number_channels, number_channels, stride=1, activation='relu'))
        self.convgru = GenGateGRU(return_sequences=True, time_steps=time_steps)  
        self.upsampling = nn.ModuleList()
        for i in range(3):
            if i > 0:
                self.upsampling.append(TimeDistributed(nn.UpsamplingBilinear2d(scale_factor=2)))
            self.upsampling.append(ResidualBlockRNN(number_channels, number_channels, stride=1, activation='leaky_relu'))          
        self.conv2 = TimeDistributed(nn.Conv2d(number_channels, 1, kernel_size=(3,3), padding=1))

    
    def forward(self, low_res):   
        initial_state = self.initialize(low_res[:,0,...])
        xt = self.conv1(low_res)
        for layer in self.res_blocks:
            xt = layer(xt)
        x = self.convgru([xt, initial_state])
        h = x[:,-1,...]
        for layer in self.upsampling:
            x = layer(x)           
        img_out = self.conv2(x)   
       
        return img_out
    
class ConvGRUGeneratorDetCon(nn.Module):    
    def __init__(self, number_channels=64, number_residual_blocks=3, upsampling_factor=2, time_steps=3):
        super(ConvGRUGeneratorDetCon, self).__init__()  
        self.initialize = InitialStateDet()
        self.conv1 = TimeDistributed(nn.Conv2d(1, number_channels, kernel_size=(3,3), padding=1))
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlockRNN(number_channels, number_channels, stride=1, activation='relu'))
        self.convgru = GenGateGRU(return_sequences=True, time_steps=time_steps)  
        self.upsampling = nn.ModuleList()
        for i in range(3):
            if i > 0:
                self.upsampling.append(TimeDistributed(nn.UpsamplingBilinear2d(scale_factor=2)))
            self.upsampling.append(ResidualBlockRNN(number_channels, number_channels, stride=1, activation='leaky_relu'))          
        self.conv2 = TimeDistributed(nn.Conv2d(number_channels, 1, kernel_size=(3,3), padding=1))
        self.constraint = SoftmaxConstraintsTime(upsampling_factor=4)
    
    def forward(self, low_res):   
        initial_state = self.initialize(low_res[:,0,...])
        xt = self.conv1(low_res)
        for layer in self.res_blocks:
            xt = layer(xt)
        x = self.convgru([xt, initial_state])
        h = x[:,-1,...]
        for layer in self.upsampling:
            x = layer(x)           
        img_out = self.conv2(x)   
        img_out = self.constraint(img_out, low_res)
        return img_out

    
class FrameConvGRU(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=3):
        super(FrameConvGRU, self).__init__()
        self.initialize = InitialStateDet()
        self.conv1 = TimeDistributed(nn.Conv2d(1, number_channels, kernel_size=(3,3), padding=1))
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlockRNN(number_channels, number_channels, stride=1, activation='relu'))
        self.convgru = GenGateGRU(return_sequences=True, time_steps=5) 
        self.conv2 = TimeDistributed(nn.Conv2d(number_channels, 1, kernel_size=(3,3), padding=1))#
        
    def forward(self, x):
        init_state = self.initialize(x[:,0,...])
        xt = self.conv1(x)
        for layer in self.res_blocks:
            xt = layer(xt)
        x = self.convgru([xt, init_state])          
        out = self.conv2(x)   
        
        return out
    
    
###############
#Deep Voxel Flow
################


def meshgrid(height, width):
    x_t = torch.matmul(
        torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
    y_t = torch.matmul(
        torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    return grid_x, grid_y


class VoxelFlow(nn.Module):

    def __init__(self):
        super(VoxelFlow, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2_bn = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(256)

        self.bottleneck = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(256)

        self.deconv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(256)

        self.deconv2 = nn.Conv2d(384, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(128)

        self.deconv3 = nn.Conv2d(192, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv3_bn = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = x[...,0,:,:]
        input_var = x
        input_size = tuple(x.size()[2:4])
        #print(x.shape)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        conv1 = self.relu(x)
        x = self.pool(conv1)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        conv2 = self.relu(x)
        x = self.pool(conv2)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        conv3 = self.relu(x)
        x = self.pool(conv3)
        x = self.bottleneck(x)
        x = self.bottleneck_bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.deconv1(x)
        x = self.deconv1_bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.deconv2(x)
        x = self.deconv2_bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.deconv3(x)
        x = self.deconv3_bn(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        flow = x[:, 0:2, :, :]
        mask = x[:, 2:3, :, :]
        grid_x, grid_y = meshgrid(input_size[0], input_size[1])
        with torch.cuda.device(input_var.get_device()):
            grid_x = torch.autograd.Variable(grid_x.repeat([input_var.size()[0], 1, 1])).cuda()
            grid_y = torch.autograd.Variable(grid_y.repeat([input_var.size()[0], 1, 1])).cuda()
        flow = 0.5 * flow    
        coor_x_1 = grid_x - flow[:, 0, :, :]
        coor_y_1 = grid_y - flow[:, 1, :, :]
        coor_x_2 = grid_x + flow[:, 0, :, :]
        coor_y_2 = grid_y + flow[:, 1, :, :]     
        output_1 = torch.nn.functional.grid_sample(input_var[:, 0:1, :, :],torch.stack([coor_x_1, coor_y_1], dim=3),padding_mode='border', align_corners=True)
        output_2 = torch.nn.functional.grid_sample(input_var[:, 1:2, :, :],torch.stack([coor_x_2, coor_y_2], dim=3),padding_mode='border', align_corners=True)
        mask = 0.5 * (1.0 + mask)
        x = mask * output_1 + (1.0 - mask) * output_2
        return x.unsqueeze(1)
    
    
class TimeEndToEndModel(nn.Module):
    def __init__(self):
        super(TimeEndToEndModel, self).__init__()
        self.temporal_sr = VoxelFlow()
        self.spatial_sr = ConvGRUGeneratorDet()
        
    def forward(self, x):
        x_in = x
        x = self.temporal_sr(x)
        #print(x.shape, x_in.shape)
        x = torch.cat((x_in[:,0:1,...], x, x_in[:,1:2,...]), dim=1)
        x = self.spatial_sr(x)
        return x
    
    
##########
##CNN for frame interpolation
###########







