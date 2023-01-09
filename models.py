import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


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
        y = y.clone()
        out = self.pool(y)
        out = y*torch.kron(lr*1/out, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    
    
class AddDownscaleConstraints(nn.Module):
    def __init__(self, upsampling_factor):
        super(AddDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        y = y.clone()
        sum_y = self.pool(y)
        out =y+ torch.kron(lr-sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    
class EnforcementOperator(nn.Module):
    def __init__(self, upsampling_factor):
        super(EnforcementOperator, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        y = y.clone()
        sum_y = self.pool(y)
        diff_P_x = torch.kron(lr-sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        sigma = torch.sign(-diff_P_x)
        out =y+ diff_P_x*(sigma+y)/(sigma+torch.kron(sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda')))
        return out 
    
class SoftmaxConstraints(nn.Module):
    def __init__(self, upsampling_factor, cwindow_size, exp_factor=1):
        super(SoftmaxConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=cwindow_size)
        self.lr_pool = torch.nn.AvgPool2d(kernel_size=int(cwindow_size/upsampling_factor))
        self.upsampling_factor = upsampling_factor
        self.cwindow_size = cwindow_size
        self.exp_factor = exp_factor
    def forward(self, y, lr):
        y = torch.exp(y*self.exp_factor)
        sum_y = self.pool(y)
        lr_sum = self.lr_pool(lr)
        out = y*torch.kron(lr_sum*1/sum_y, torch.ones((self.cwindow_size,self.cwindow_size)).to('cuda'))
        return out
    


class MultIn(nn.Module):
    def __init__(self, factor):
        super(MultIn, self).__init__()
        self.factor = factor
    def forward(self, y, lr):
        return y*lr*self.factor
    
class AddChannels(nn.Module):
    def __init__(self):
        super(AddChannels, self).__init__()
    def forward(self, y):
        return torch.sum(y, dim=1).unsqueeze(1)
         
        
class ResNet(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, constraints='none', dim=1, cwindow_size=4):
        super(ResNet, self).__init__()
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
            self.constraints = SoftmaxConstraints(upsampling_factor=upsampling_factor, cwindow_size=cwindow_size)
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
            
        self.dim = dim    
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
            out = self.conv1(x[:,0,...])
            for layer in self.upsampling:
                out = layer(out)
            out = self.conv2(out)    
            for layer in self.res_blocks:
                out = layer(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if self.is_constraints:
                out[:,...] = self.constraints(out, x[:,0,...])
            out = out.unsqueeze(1)
            return out
        
class ResNet3(nn.Module):
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, constraints='none', dim=1, cwindow_size=2):
        super(ResNet3, self).__init__()
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
        for k in range(1):
            self.upsampling.append(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=3, padding=0, stride=3) )
        # Next layer after upper sampling
        self.conv3 = nn.Sequential(nn.Conv2d(number_channels, number_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True))
        # Final output layer
        self.conv4 = nn.Conv2d(number_channels, dim, kernel_size=1, stride=1, padding=0)      
        #optional renomralization layer
        self.is_constraints = False
        if constraints == 'softmax':
            self.constraints = SoftmaxConstraints(upsampling_factor=upsampling_factor, cwindow_size=cwindow_size)
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
            
        self.dim = dim    
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
            out = self.conv1(x[:,0,...])
            for layer in self.upsampling:
                out = layer(out)
            out = self.conv2(out)    
            for layer in self.res_blocks:
                out = layer(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if self.is_constraints:
                out[:,...] = self.constraints(out, x[:,0,...])
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
    
class MultDownscaleConstraintsTime(nn.Module):
    def __init__(self, upsampling_factor):
        super(MultDownscaleConstraintsTime, self).__init__()
        self.pool = TimeDistributed(torch.nn.AvgPool2d(kernel_size=upsampling_factor))
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        y = y.clone()
        sum_y = self.pool(y)
        out = y*torch.kron(lr*1/sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    
    
class AddDownscaleConstraintsTime(nn.Module):
    def __init__(self, upsampling_factor):
        super(AddDownscaleConstraintsTime, self).__init__()
        self.pool = TimeDistributed(torch.nn.AvgPool2d(kernel_size=upsampling_factor))
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        y = y.clone()
        sum_y = self.pool(y)
        out =y+ torch.kron(lr-sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    
class EnforcementOperatorTime(nn.Module):
    def __init__(self, upsampling_factor):
        super(EnforcementOperatorTime, self).__init__()
        self.pool = TimeDistributed(torch.nn.AvgPool2d(kernel_size=upsampling_factor))
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        y = y.clone()
        sum_y = self.pool(y)
        diff_P_x = torch.kron(lr-sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        sigma = torch.sign(-diff_P_x)
        out =y+ diff_P_x*(sigma+y)/(sigma+torch.kron(sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda')))
        return out 
    
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
    def __init__(self, return_sequences=True, time_steps=3):
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
        
        
class ConvGRUGeneratorDet(nn.Module):    
    def __init__(self, number_channels=64, number_residual_blocks=3, upsampling_factor=2, time_steps=3, constraints='none', cwindow_size=2):
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
        self.is_constraints = False
        if constraints == 'softmax':
            self.constraints = SoftmaxConstraintsTime(upsampling_factor=upsampling_factor)
            self.is_constraints = True
        elif constraints == 'enforce_op':
            self.constraints = EnforcementOperatorTime(upsampling_factor=upsampling_factor)
            self.is_constraints = True
        elif constraints == 'add':
            self.constraints = AddDownscaleConstraintsTime(upsampling_factor=upsampling_factor)
            self.is_constraints = True
        elif constraints == 'mult':
            self.constraints = MultDownscaleConstraintsTime(upsampling_factor=upsampling_factor)
            self.is_constraints = True
    
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
        if self.is_constraints:
            img_out = self.constraints(img_out, low_res)
        return img_out

    
    
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
    def __init__(self, number_channels=64, number_residual_blocks=3, upsampling_factor=2, time_steps=3, constraints='none'):
        super(TimeEndToEndModel, self).__init__()
        self.temporal_sr = VoxelFlow()
        self.spatial_sr = ConvGRUGeneratorDet( number_channels=number_channels, number_residual_blocks=number_residual_blocks, upsampling_factor=upsampling_factor, time_steps=3, constraints=constraints)       
    def forward(self, x):
        x_in = x
        x = self.temporal_sr(x)
        x = torch.cat((x_in[:,0:1,...], x, x_in[:,1:2,...]), dim=1)
        x = self.spatial_sr(x)
        return x
    














