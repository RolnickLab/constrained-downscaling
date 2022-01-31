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
    mean = target_train.mean()
    std = target_train.std()
    max_val = target_train.max()

    input_train = (input_train - mean)/std
    target_train = (target_train - mean)/std
    input_val = (input_val - mean)/std
    target_val = (target_val - mean)/std
    train_data = TensorDataset(input_train, target_train)
    val_data = TensorDataset(input_val, target_val)
    train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    return [train, val, mean, std, max_val]

def load_model(args, discriminator=False):
    if discriminator:
        model = models.Discriminator()
    else:
        model = models.ResNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, downscale_constraints=args.downscale_constraints,  softmax_constraints=args.softmax_constraints)
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
    inputs.unsqueeze_(1)
    targets.unsqueeze_(1)
    return inputs, targets

def process_for_eval(inputs, targets, mean, std): 
    inputs = inputs*std+mean           
    targets = targets*std+mean
    return inputs, targets

def is_gan(args):
    if args.model == 'gan':
        return True
    else: 
        return False
    
def is_noisegan(args):
    return is_gan.args and args.noise
    
    



    
    
