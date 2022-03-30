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
    mean = torch.zeros((args.dim))
    std = torch.zeros((args.dim))
    max_val = torch.zeros((args.dim))
    if len(input_train.shape)==3:
        input_train = torch.unsqueeze(input_train, dim=1)
        target_train = torch.unsqueeze(target_train, dim=1)
        input_val = torch.unsqueeze(input_val, dim=1)
        target_val = torch.unsqueeze(target_val, dim=1)
        #mid_train = torch.unsqueeze(mid_train, dim=1)
        #mid_val = torch.unsqueeze(mid_val, dim=1)
    for i in range(args.dim):
        mean[i] = target_train[:,i,:,:].mean()
        std[i] = target_train[:,i,:,:].std()
        max_val[i] = target_train[:,i,:,:].max()
    print(mean, std)
    if args.scale == 'standard':
        print(input_train.shape)
        for i in range(args.dim):
            
            input_train[:,i,:,:] = (input_train[:,i,:,:] - mean[i])/std[i]
            target_train[:,i,:,:] = (target_train[:,i,:,:] - mean[i])/std[i]
            input_val[:,i,:,:] = (input_val[:,i,:,:] - mean[i])/std[i]
            target_val[:,i,:,:] = (target_val[:,i,:,:] - mean[i])/std[i]
            #mid_train[:,i,:,:] = (mid_train[:,i,:,:] - mean[i])/std[i]
            #mid_val[:,i,:,:] = (mid_val[:,i,:,:] - mean[i])/std[i]
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
        model = models.Discriminator()
    else:
        if args.noise:
            model = models.ResNetNoise(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, downscale_constraints=args.downscale_constraints,  softmax_constraints=args.softmax_constraints, dim=args.dim)
        elif args.model == 'mr_constr':
            model = models.MRResNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, downscale_constraints=args.downscale_constraints,  softmax_constraints=args.softmax_constraints, dim=args.dim)
        elif args.model == 'conv_gru':
            model = models.ConvGRUGeneratorDet()
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

def process_for_eval(inputs, targets, mean, std, max_val, args): 
    if args.scale == 'standard':
        for i in range(args.dim):
            inputs[:,i,:,:] = inputs[:,i,:,:]*std[i]+mean[i]
            targets[:,i,:,:] = targets[:,i,:,:]*std[i]+mean[i]
          
    elif args.scale == 'minmax':
        inputs = inputs*max_val.to(device)          
        targets = targets*max_val.to(device)
    elif args.scale == 'log':
        inputs = torch.exp(inputs)-1
        targets = torch.exp(targets)-1
    return inputs, targets

def is_gan(args):
    if args.model == 'gan':
        return True
    else: 
        return False
    
def is_noisegan(args):
    return is_gan.args and args.noise
    
    



    
    
