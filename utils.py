import torch
import torch.optim as optim
import torch.nn as nn
import models
import Learnable_basis
from torch.utils.data import DataLoader, TensorDataset
device = 'cuda'

def load_data(args):
    input_train = torch.load('./data/train/'+args.dataset+'/input_train.pt')
    #target_train = torch.load('./data/train/dataset28/target_train.pt')
    target_train = torch.load('./data/train/'+args.dataset+'/target_train.pt')
    #input_train = input_train[:4000,...]
    #target_train = target_train[:4000,...]
    if args.test:
        input_val = torch.load('./data/test/'+args.dataset+'/input_test.pt')
        target_val = torch.load('./data/test/'+args.dataset+'/target_test.pt')
    else:
        input_val = torch.load('./data/val/'+args.dataset+'/input_val.pt')
        target_val = torch.load('./data/val/'+args.dataset+'/target_val.pt')
        #target_val = torch.load('./data/val/'+args.dataset+'/target_val.pt')
    #define dimesions
    global train_shape_in , train_shape_out, val_shape_in, val_shape_in
    train_shape_in = input_train.shape
    train_shape_out = target_train.shape
    val_shape_in = input_val.shape
    val_shape_out = target_val.shape
    #mean, std, max
    mean = target_train.mean()
    std = target_train.std()
    global max_val, min_val
   
    max_val = torch.zeros((args.dim_channels,1))
    min_val = torch.zeros((args.dim_channels,1))
    for i in range(args.dim_channels):
        max_val[i] = target_train[:,0,i,...].max()
        min_val[i] = target_train[:,0,i,...].min()
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
        for i in range(args.dim_channels):
            input_train[:,0,i,...] = (input_train[:,0,i,...]-min_val[i]) /(max_val[i]-min_val[i])
            target_train[:,0,i,...] = (target_train[:,0,i,...] -min_val[i])/(max_val[i]-min_val[i])
            input_val[:,0,i,...] = (input_val[:,0,i,...]-min_val[i])/(max_val[i]-min_val[i])
            target_val[:,0,i,...] = (target_val[:,0,i,...]-min_val[i])/(max_val[i]-min_val[i])
    elif args.scale == 'minmax_fixed':
        input_train = input_train /args.max
        target_train = target_train /args.max
        input_val = input_val/args.max
        target_val = target_val/args.max
    elif args.scale == 'minus_to_one_fixed':
        input_train = 2*input_train /args.max-1
        target_train = 2*target_train /args.max -1
        input_val = 2*input_val/args.max -1
        target_val = 2*target_val/args.max -1
    elif args.scale == 'log':
        input_train = torch.log(input_train+1)
        target_train = torch.log(target_train+1)
        input_val = torch.log(input_val+1)
        target_val = torch.log(target_val+1)  
    print(input_train.mean(), input_train.std(), input_train.max(), input_train.min())
    print(target_train.mean(), target_train.std(), target_train.max(), target_train.min())
    print(input_val.mean(), input_val.std(), input_val.max(), input_val.min())
    print(target_val.mean(), target_val.std(), target_val.max(), target_val.min())
    train_data = TensorDataset(input_train,  target_train)
    val_data = TensorDataset(input_val, target_val)
    train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) 
    val = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    return [train, val, mean, std, max_val, train_shape_in, train_shape_out, val_shape_in, val_shape_out]

def load_model(args, discriminator=False):

    if discriminator:
        #if args.time:
        #    model = models.DiscriminatorRNN()
        #else:
        model = models.Discriminator()
    else:
        if args.noise:
            if args.model == 'conv_gru':
                model = models.ConvGRUGenerator()
            elif args.model == 'gan_4x':
                model = models.ResNet2UpNoise(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints, dim=1, output_mr=args.mr)
            else:
                model = models.ResNetNoise(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints)
        elif args.model == 'mr_constr':
            model = models.MRResNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, downscale_constraints=args.downscale_constraints,  softmax_constraints=args.softmax_constraints, dim=1, output_mr=args.mr)
        elif args.model == 'conv_gru_det':
            model = models.ConvGRUGeneratorDet( number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, time_steps=3, constraints=args.constraints)
        elif args.model == 'conv_gru_det_con':
            model = models.ConvGRUGeneratorDetCon()
        elif args.model == 'frame_gru':
            model = models.FrameConvGRU()
        elif args.model == 'voxel_flow':
            model = models.VoxelFlow()
        elif args.model == 'time_end_to_end':
            model = models.TimeEndToEndModel( number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, time_steps=3, constraints=args.constraints)
        elif args.model == 'resnet_new':
            model = models.CNNUp()
        elif args.model == 'esrgan':
            model = models.ESRGANGenerator()
        elif args.model == 'srgan':
            model = models.SRGANGenerator(n_residual_blocks=args.number_residual_blocks, upsample_factor=args.upsampling_factor)
        elif args.model == 'res_2up':
            model = models.ResNet2Up(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints, dim=1, output_mr=args.mr)
        elif args.model == 'res_4up':
            model = models.ResNet4Up(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints, dim=1, output_mr=args.mr)
        elif args.model == 'res_3up':
            model = models.ResNet3Up()
            #model = models.ResNet3Up(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints, dim=1, output_mr=args.mr)
        elif args.model == 'motifnet':
            model = models.MotifNet(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints, dim=1)
        elif args.model == 'motifnet_learnable':
            model = models.MotifNetLearnBasis(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints, dim=1, l2_reg=args.l2_reg)
        elif args.model == 'mixture':
            model = models.MixtureModel(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints, dim=1)
        elif args.model == 'gan':
            model = models.ResNet2(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints, dim=args.dim_channels)
        else:
            model = models.ResNet2(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=args.upsampling_factor, noise=args.noise, constraints=args.constraints, dim=args.dim_channels)
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

def mass_loss(output, true_value, args):
    ds_out = torch.nn.functional.avg_pool2d(output[:,0,0,:,:], args.upsampling_factor)
    ds_true = torch.nn.functional.avg_pool2d(true_value[:,0,0,:,:], args.upsampling_factor)
    return torch.nn.functional.mse_loss(ds_out, ds_true)

def get_loss(output, true_value, args):
    if args.loss == 'mass_constraints':
        return args.alpha*torch.nn.functional.mse_loss(output, true_value) + (1-args.alpha)*mass_loss(output, true_value, args)
    else:
        return torch.nn.functional.mse_loss(output, true_value)
    
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
        for i in range(args.dim_channels):
            outputs[:,0,i,...] = outputs[:,0,i,...]*(max_val[i].to(device)-min_val[i].to(device))+min_val[i].to(device) 
            targets[:,0,i,...] = targets[:,0,i,...]*(max_val[i].to(device)-min_val[i].to(device))+min_val[i].to(device)
    elif args.scale == 'minmax_fixed':
        outputs = outputs*args.max         
        targets = targets*args.max
    elif args.scale == 'minus_to_one_fixed':
        outputs = (outputs+1)/2*args.max         
        targets = (targets+1)/2*args.max
    elif args.scale == 'log':
        outputs = torch.exp(inputs)-1
        targets = torch.exp(targets)-1
    return outputs, targets

def is_gan(args):
    if args.model == 'gan':
        return True
    if args.model == 'gan_4x':
        return True
    elif args.model == 'conv_gru':
        return True
    else: 
        return False
    
def is_noisegan(args):
    return is_gan.args and args.noise



    
