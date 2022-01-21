import torch
import models
device = 'cuda'

def load_data(args):
    train_data = torch.load('./data/train/'+args.dataset+'.pt')
    val_data = torch.load('./data/val/'+args.dataset_name+'.pt')
    mean = train_data[:][1].mean()
    std = train_data[:][1].std()
    train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    return [train, val, mean, std]

def transform_data(data, args):
    train_data = data[0]
    val_data = data[1]
    train_data[:][0] = (train_data[:][0] - data[2])/data[3]
    train_data[:][1] = (train_data[:][1] - data[2])/data[3]
    val_data[:][0] = (val_data[:][0] - data[2])/data[3]
    val_data[:][1] = (val_data[:][1] - data[2])/data[3]
    return [train_data, val_data, data[2], data[3]]

def load_model(args, discriminator=False):
    if discriminator:
        model = models.Discriminator()
    else:
        model = models.ResNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, renorm=args.renorm)
    model.to(device)
    return model

def get_optimizer(args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def get_criterion(args, discriminator=False):
    if discriminator:
        nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    return criterion
    
def process_for_training(inputs, targets): 
    inputs = inputs.to(device)            
    targets = targets.to(device)
    inputs.unsqueeze_(1)
    targets.unsqueeze_(1)
    return inputs, targets
    



    
    
