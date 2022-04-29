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
    #define dimesions
    global train_shape_in , train_shape_out, val_shape_in, val_shape_in
    train_shape_in = input_train.shape
    train_shape_out = target_train.shape
    val_shape_in = input_val.shape
    val_shape_out = target_val.shape
    #mean, std, max
    mean = target_train.mean()
    std = target_train.std()
    max_val = target_train.max()
    #transform data
    if args.scale == 'standard':          
        input_train = (input_train - mean)/std
        input_val = (input_val - mean)/std
        target_train = (target_train - mean)/std
        target_val = (target_val - mean)/std
    elif args.scale == 'standard_fixed':
        input_train = (input_train - args.mean)/args.std
        input_val = (input_val - args.mean)/args.std
        target_train = (target_train - args.mean)/args.std
        target_val = (target_val - args.mean)/args.std  
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
    return inputs,  targets

def process_for_eval(outputs, targets, mean, std, max_val, args): 
    if args.scale == 'standard':
        outputs = outputs*std+mean
        targets = targets*std+mean
    elif args.scale == 'standard_fixed':
         outputs = outputs*args.std+args.mean
        targets = targets*args.std+args.mean 
    elif args.scale == 'minmax':
        outputs = outputs*max_val         
        targets = targets*max_val
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



    
