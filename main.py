from training import run_training
from utils import load_data
import numpy as np
import argparse

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dataset28', help="choose a data set to use")
    parser.add_argument("--scale", default='minmax_fixed', help="standard, minmax, none")
    parser.add_argument("--model", default='res_2up')
    parser.add_argument("--model_id", default='nu_aaai_symp_cnn_softmaxsingle_long_4_3')
    parser.add_argument("--number_channels", default=64, type=int)
    parser.add_argument("--number_residual_blocks", default=4, type=int)#""!!change
    parser.add_argument("--upsampling_factor", default=4, type=int)
    parser.add_argument("--noise", default=False)
    parser.add_argument("--constraints", default='softmax_single')
    parser.add_argument("--lr", default=0.001, help="learning rate", type=float)
    parser.add_argument("--loss", default='mse')
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--weight_decay", default=1e-9, type=float)
    parser.add_argument("--batch_size", default=256, type=int) #!!change
    parser.add_argument("--epochs", default=200, type=int)
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
    parser.add_argument("--test", default=False, type=bool)
    
    
    return parser.parse_args()

def main(args):
    #load data
    data = load_data(args)
    
    #run training
    run_training(args, data)    
    
if __name__ == '__main__':
    args = add_arguments()
    main(args)