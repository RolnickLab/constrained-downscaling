import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#torch.set_default_dtype(torch.float64)
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
    
    
class AddDownscaleConstraints(nn.Module):
    def __init__(self, upsampling_factor):
        super(AddDownscaleConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
    def forward(self, y, lr):
        sum_y = self.pool(y)
        out =y+ torch.kron(lr-sum_y, torch.ones((self.upsampling_factor,self.upsampling_factor)).to('cuda'))
        return out
    
    
class SoftmaxConstraints(nn.Module):
    def __init__(self, upsampling_factor, exp_factor=1):
        super(SoftmaxConstraints, self).__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=upsampling_factor)
        self.upsampling_factor = upsampling_factor
        self.exp_factor = exp_factor
    def forward(self, y, lr):
        y = torch.exp(self.exp_factor*y)
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
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2, noise=False, downscale_constraints=False, softmax_constraints=False, dim=1):
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
            for layer in self.upsampling:
                out = layer(out)
            out = self.conv2(out)    
            for layer in self.res_blocks:
                out = layer(out)
            out = self.conv3(out)
            out = self.conv4(out)
            if self.downscale_constraints:
                out = self.downscale_constraint(out, x)
            #out[:,0,:,:] *= 16
            return out  
          
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
        out = torch.reshape(out, (x.shape))
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
    
####time-series model
###work in progress

'''

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class CustomGateGRU(nn.Module):
    def __init__(self, update_gate=None, reset_gate=None, output_gate=None, return_sequences=False, time_steps=1):
        super(CustomGateGRU).__init__()
        self.update_gate = update_gate
        self.reset_gate = reset_gate
        self.output_gate = output_gate
        self.return_sequences = return_sequences
        self.time_steps = time_steps

    def forward(self, inputs):
        (xt,h) = inputs
        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            xh = torch.cat((x,h), dim=-1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(torch.cat((x,r*h), dim=-1))
            h = z*h + (1-z)*torch.tanh(o)
            if self.return_sequences:
                h_all.append(h)
                
def gen_gate(activation='sigmoid'):
    def gate(x):
        x = nn.ReflectionPad2d(padding=(1,1))(x)
        x = nn.Connv2d(256-noise_in_update.shape[-1], kernel_size=(3,3))(x)
        if activation is not None:
            x = nn.Sigmoid(x)
        return x
    return Lambda(gate)
                

class ConvGRUGenerator(nn.Module):    
    def __init__(self, number_channels=64, number_residual_blocks=4, upsampling_factor=2):
    super(ConvGRUGenerator, self).__init__()  
        self.reflpadd1 = nn.ReflectionPad2d(padding=(1,1))
        self.conv1 = nn.Connv2d(num_channels, kernel_size=(3,3))
        self.res_blocks = nn.ModuleList()
        for k in range(number_residual_blocks):
            self.res_blocks.append(ResidualBlock(number_channels, number_channels))
        self.convgru = CustomGateGRU(update_gate=gen_gate(), reset_gate=gen_gate(), output_gate=gen_gate(activation=None), return_sequences=True, time_steps=num_timesteps)
        self.upsampling = nn.ModuleList()
        for k in range(int(np.rint(np.log2(upsampling_factor)))+1):
            if k > 0:
                self.upsampling.append(TimeDistributed(nn.ConvTranspose2d(number_channels, number_channels, kernel_size=2, padding=0, stride=2)))
            self.upsampling.append(ResidualBlock(number_channels, number_channels))
        self.reflpadd2 = nn.ReflectionPad2d(padding=(1,1))
        self.conv2 = nn.Connv2d(num_channels, kernel_size=(3,3)) 
    
    def forward(self, x):
        xt = TimeDistributed(self.reflpadd1)(low_res)
        xt = TimeDistributed(self.conv1)(xt)
        xt = torch.cat((xt, noise_in_update))
        for layer in self.res_blocks:
            xt = layer(xt)
        x = self.convgru([xt, initial_state])
        h = x[:,-1,...]
        for layer in self.upsampling:
            x = layer(x)
            
        x = TimeDistributed(self.reflpadd2)(x)
        img_out = TimeDistributed(self.conv2)(x)
        
        return [img_out, h]
        

def ConvGRUGenerator(num_channels=1, num_timesteps=8, num_preproc=3):
    initial_state = Input(shape=(None,None,256))
    noise_in_update = Input(shape=(num_timesteps,None,None,8),
        name="noise_in_update")
    lores_in = Input(shape=(num_timesteps,None,None,num_channels),
        name="cond_in")
    inputs = [lores_in, initial_state, noise_in_update]

    xt = TimeDistributed(ReflectionPadding2D(padding=(1,1)))(lores_in)
    xt = TimeDistributed(Conv2D(256-noise_in_update.shape[-1], 
        kernel_size=(3,3)))(xt)
    xt = Concatenate()([xt,noise_in_update])
    for i in range(num_preproc):
        xt = res_block(256, time_dist=True, activation='relu')(xt)

    def gen_gate(activation='sigmoid'):
        def gate(x):
            x = ReflectionPadding2D(padding=(1,1))(x)
            x = Conv2D(256, kernel_size=(3,3))(x)
            if activation is not None:
                x = Activation(activation)(x)
            return x
        return Lambda(gate)
    
    x = CustomGateGRU(
        update_gate=gen_gate(),
        reset_gate=gen_gate(),
        output_gate=gen_gate(activation=None),
        return_sequences=True,
        time_steps=num_timesteps
    )([xt,initial_state])

    h = x[:,-1,...]
    
    block_channels = [256, 256, 128, 64, 32]
    for (i,channels) in enumerate(block_channels):
        if i > 0:
            x = TimeDistributed(UpSampling2D(interpolation='bilinear'))(x)
        x = res_block(channels, time_dist=True, activation='leakyrelu')(x)

    x = TimeDistributed(ReflectionPadding2D(padding=(1,1)))(x)
    img_out = TimeDistributed(Conv2D(num_channels, kernel_size=(3,3),
        activation='sigmoid'))(x)

    model = Model(inputs=inputs, outputs=[img_out,h])

    def noise_shapes(img_shape=(128,128)):
        noise_shape_update = (
            num_timesteps, img_shape[0]//16, img_shape[1]//16, 8
        )
        return [noise_shape_update]

    return (model, noise_shapes)

def initial_state_model(num_preproc=3):
    initial_frame_in = Input(shape=(None,None,1))
    noise_in_initial = Input(shape=(None,None,8),
        name="noise_in_initial")

    h = ReflectionPadding2D(padding=(1,1))(initial_frame_in)
    h = Conv2D(256-noise_in_initial.shape[-1], kernel_size=(3,3))(h)
    h = Concatenate()([h,noise_in_initial])
    for i in range(num_preproc):
        h = res_block(256, activation='relu')(h)

    return Model(
        inputs=[initial_frame_in,noise_in_initial],
        outputs=h
    )

def generator_initialized(gen, init_model,
    num_channels=1, num_timesteps=8):
    noise_in_initial = Input(shape=(None,None,8),
        name="noise_in_initial")
    noise_in_update = Input(shape=(num_timesteps,None,None,8),
        name="noise_in_update")
    lores_in = Input(shape=(num_timesteps,None,None,num_channels),
        name="cond_in")
    inputs = [lores_in, noise_in_initial, noise_in_update]

    initial_state = init_model([lores_in[:,0,...], noise_in_initial])
    (img_out,h) = gen([lores_in, initial_state, noise_in_update])

    model = Model(inputs=inputs, outputs=img_out)

    def noise_shapes(img_shape=(128,128)):
        noise_shape_initial = (img_shape[0]//16, img_shape[1]//16, 8)
        noise_shape_update = (
            num_timesteps, img_shape[0]//16, img_shape[1]//16, 8
        )
        return [noise_shape_initial, noise_shape_update]

    return (model, noise_shapes)


def generator_deterministic(gen_init, num_channels=1, num_timesteps=8):
    lores_in = Input(shape=(num_timesteps,None,None,num_channels),
        name="cond_in")

    def zeros_noise(input, which):
        shape = tf.shape(input)
        if which == 'init':
            shape = tf.stack([shape[0],shape[1],shape[2],8])
        elif which == 'update':
            shape = tf.stack([shape[0],num_timesteps,shape[1],shape[2],8])
        return tf.fill(shape, 0.0)

    init_zeros = Lambda(lambda x: zeros_noise(x, 'init'))(lores_in)
    update_zeros = Lambda(lambda x: zeros_noise(x, 'update'))(lores_in)
    img_out = gen_init([lores_in, init_zeros, update_zeros])

    model = Model(inputs=lores_in, outputs=img_out)

    return model
    
    '''



