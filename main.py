from training import run_training
from utils import load_data
import numpy as np
import argparse
import os
import torch

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="era5_twc", help="choose a data set to use")
    parser.add_argument("--model", default="resnet2")
    parser.add_argument("--model_id", default="test")
    parser.add_argument("--upsampling_factor", default=4, type=int)
    parser.add_argument("--constraints", default="none") 
    parser.add_argument("--number_channels", default=32, type=int)
    parser.add_argument("--number_residual_blocks", default=4, type=int)
    parser.add_argument("--lr", default=0.001, help="learning rate", type=float)
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--weight_decay", default=1e-9, type=float)
    parser.add_argument("--batch_size", default=64, type=int) 
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--alpha", default=0.99, type=float)
    parser.add_argument("--test_val_train", default="val")
    parser.add_argument("--training_evalonly", default="training")
    parser.add_argument("--dim_channels", default=1, type=int)
    parser.add_argument("--adv_factor", default=0.0001, type=float)
    return parser.parse_args()

def main(args):
    #load data
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./data/prediction'):
        os.makedirs('./data/prediction')
    if args.training_evalonly == 'training':
        data = load_data(args)
        #run training
        run_training(args, data)    
    else:       
        data = load_data(args)
        #run training
        evaluate_model(data, args)
        
if __name__ == '__main__':
    args = add_arguments()
    main(args)