from training import run_training
from utils import load_data, transform_data
import argparse

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dataset0', help="choose a data set to use")
    parser.add_argument("--scale", default='standard', help="standard,minmax, non")
    parser.add_argument("--model", default='gan')
    parser.add_argument("--model_id", default='gan_1')
    parser.add_argument("--number_channels", default=64)
    parser.add_argument("--number_residual_blocks", default=4)
    parser.add_argument("--upsampling_factor", default=2)
    parser.add_argument("--noise", default=False)
    parser.add_argument("--renorm", default=False)
    parser.add_argument("--lr", default=0.001, help="learning rate")
    parser.add_argument("--loss", default='mse')
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--weight_decay", default=1e-9)
    parser.add_argument("--batch_size", default=256)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--early_stop", default=True)
    return parser.parse_args()

def main():
    #load data
    data = load_data(args)
    
    #transform data
    data = transform_data(data, args)
    
    #run training
    run_training(data, args)    
    
if __name__ == '__main__':
    args = add_arguments()
    main(args)