from training import run_training
from utils import load_data
import numpy as np
import argparse
import torch

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dataset34', help="choose a data set to use")
    parser.add_argument("--scale", default='minmax', help="standard, minmax, none")
    parser.add_argument("--model", default='motifnet_learnable')
    parser.add_argument("--model_id", default='motifbased_exp_ood_softmaxfirst_test')
    parser.add_argument("--number_channels", default=32, type=int)
    parser.add_argument("--number_residual_blocks", default=4, type=int)#""!!change
    parser.add_argument("--upsampling_factor", default=4, type=int)
    parser.add_argument("--noise", default=False)
    parser.add_argument("--constraints", default='none') #softmax sum
    parser.add_argument("--lr", default=0.001, help="learning rate", type=float)
    parser.add_argument("--loss", default='mse')
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--weight_decay", default=1e-9, type=float)
    parser.add_argument("--batch_size", default=64, type=int) #!!change
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--reg_factor", default=1, type=int)
    parser.add_argument("--adv_factor", default=0.0001, type=float)
    parser.add_argument("--early_stop", default=False,  type=bool)
    parser.add_argument("--time", default=False,  type=bool)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--exp_factor", default=1, type=float)
    parser.add_argument("--mean", default=21.4, type=float)
    parser.add_argument("--std", default=17.3, type=float)
    parser.add_argument("--max", default=135, type=float)
    parser.add_argument("--mr", default=False, type=bool)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--test_val_train", default='val')
    parser.add_argument("--l2_reg", default=False, type=bool)
    parser.add_argument("--dim_channels", default=1, type=int)
    parser.add_argument("--constraints_window_size", default=4, type=int)
    parser.add_argument("--ensemble", default=False)
    parser.add_argument("--factor", default=4, type=int)
    parser.add_argument("--nn", default=True)
    #parser.add_argument("--test", default=True)
    parser.add_argument("--time_steps", type=int, default=1)
    parser.add_argument("--mass_violation", type=bool, default=True)
    parser.add_argument("--time_sr", default=False)
    parser.add_argument("--save_mass_loss", default=False)
    
    return parser.parse_args()

def main(args):
    #load data
    #torch.cuda.empty_cache()
    data = load_data(args)
    print(args.epochs)
    
    #run training
    run_training(args, data)    
    
if __name__ == '__main__':
    args = add_arguments()
    main(args)