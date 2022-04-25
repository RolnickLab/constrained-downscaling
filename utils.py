import torch
import torch.optim as optim
import torch.nn as nn
import models
from torch.utils.data import DataLoader, TensorDataset
device = 'cuda'

def load_data(args):
    input_train = torch.load('./data/train/'+args.dataset+'/input_train.pt')
    target_train = torch.load('./data/train/'+args.dataset+'/target_train.pt')
    input_val = torch.load('./data/val/'+args.dataset+'/input_val.pt')
    target_val = torch.load('./data/val/'+args.dataset+'/target_val.pt')
    #mid_train = torch.load('./data/train/'+args.dataset+'/mid_train.pt')
    #mid_val = torch.load('./data/val/'+args.dataset+'/mid_val.pt')
    mean = torch.zeros((args.dim_out)) #max of dims
    std = torch.zeros((args.dim_out))
    max_val = torch.zeros((args.dim_out))
    if len(input_train.shape)==3:
        input_train = torch.unsqueeze(input_train, dim=1)
        target_train = torch.unsqueeze(target_train, dim=1)
        input_val = torch.unsqueeze(input_val, dim=1)
        target_val = torch.unsqueeze(target_val, dim=1)
        #mid_train = torch.unsqueeze(mid_train, dim=1)
        #mid_val = torch.unsqueeze(mid_val, dim=1)
    '''
    elif len(input_train.shape)==3 and args.time:
        input_train = torch.unsqueeze(input_train, dim=1)
        target_train = torch.unsqueeze(target_train, dim=1)
        input_val = torch.unsqueeze(input_val, dim=1)
        target_val = torch.unsqueeze(target_val, dim=1)'''
    for i in range(args.dim_out):
        mean[i] = 19#target_train[:,i,:,:].mean()
        std[i] = 16#target_train[:,i,:,:].std()
        max_val[i] = target_train[:,0,:,:].max()
    print(mean, std)
    if args.scale == 'standard':
           
        input_train = (input_train - mean[0])/std[0]
        input_val = (input_val - mean[0])/std[0]
        target_train = (target_train - mean[0])/std[0]
        target_val = (target_val - mean[0])/std[0]
    elif args.scale == 'minmax':
        input_train = input_train /max_val
        target_train = target_train /max_val
        input_val = input_val/max_val
        target_val = target_val/max_val
    elif args.scale == 'log':
        input_train = torch.log(input_train+1)
        target_train = torch.log(target_train+1)
        input_val = torch.log(input_val+1)
        target_val = torch.log(target_val+1)
        
    
        
    train_data = TensorDataset(input_train,  target_train)
    val_data = TensorDataset(input_val, target_val)
    train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    return [train, val, mean, std, max_val]

def load_model(args, discriminator=False):

    if discriminator:
        if args.time:
            model = models.DiscriminatorRNN()
        else:
            model = models.Discriminator()
    else:
        if args.noise:
            if args.model == 'conv_gru':
                model = models.ConvGRUGenerator()
            else:
                model = models.ResNetNoise(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, downscale_constraints=args.downscale_constraints,  softmax_constraints=args.softmax_constraints, dim=args.dim)
        elif args.model == 'mr_constr':
            model = models.MRResNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, downscale_constraints=args.downscale_constraints,  softmax_constraints=args.softmax_constraints, dim=args.dim)
        elif args.model == 'conv_gru_det':
            model = models.ConvGRUGeneratorDet()
        elif args.model == 'frame_gru':
            model = models.FrameConvGRU()
        elif args.model == 'voxel_flow':
            model = models.VoxelFlow()
        elif args.model == 'time_end_to_end':
            model = models.TimeEndToEndModel()
        
        else:
            model = models.ResNet2(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, downscale_constraints=args.downscale_constraints,  softmax_constraints=args.softmax_constraints, dim=args.dim)
    model.to(device)
    return model

def get_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def get_criterion(args, discriminator=False):
    if discriminator:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    return criterion
    
def process_for_training(inputs, targets): 
    inputs = inputs.to(device)            
    targets = targets.to(device)
   
    #inputs.unsqueeze_(1)
    #targets.unsqueeze_(1)
    return inputs,  targets

def process_for_eval(outputs, targets, mean, std, max_val, args): 
    if args.scale == 'standard':
        for i in range(args.dim_out):
            outputs[:,i,:,:] = outputs[:,i,:,:]*std[i]+mean[i]
            targets[:,i,:,:] = targets[:,i,:,:]*std[i]+mean[i]
          
    elif args.scale == 'minmax':
        outputs = outputs*max_val.to(device)          
        targets = targets*max_val.to(device)
    elif args.scale == 'log':
        outputs = torch.exp(inputs)-1
        targets = torch.exp(targets)-1
    return outputs, targets

def is_gan(args):
    if args.model == 'gan':
        
        return True
    elif args.model == 'conv_gru':
        return True
    else: 
        return False
    
def is_noisegan(args):
    return is_gan.args and args.noise



    
