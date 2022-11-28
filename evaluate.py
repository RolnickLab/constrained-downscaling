import torch
import argparse
from training import evaluate_double_model, create_report, evaluate_model
import models
from utils import load_data,load_model
from scoring import main_scoring

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dataset34', help="choose a data set to use")
    parser.add_argument("--scale", default='minmax_fixed', help="standard, minmax, none")
    parser.add_argument("--model", default='resnet2')
    parser.add_argument("--model_id", default='resnet2_ood_test')
    parser.add_argument("--model_id2", default='nu_aaai_symp_cnn_softmaxsingle_long_4_2')
    parser.add_argument("--number_channels", default=64, type=int)
    parser.add_argument("--number_residual_blocks", default=4, type=int)
    parser.add_argument("--upsampling_factor", default=4, type=int)
    parser.add_argument("--noise", default=False)
    parser.add_argument("--downscale_constraints", default=True, type=bool)
    parser.add_argument("--softmax_constraints", default=True, type=bool)
    parser.add_argument("--mr_constrained", default=False)
    parser.add_argument("--lr", default=0.001, help="learning rate", type=float)
    parser.add_argument("--loss", default='mse')
    parser.add_argument("--optimizer", default='adam')
    parser.add_argument("--weight_decay", default=1e-9, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--reg_factor", default=1, type=int)
    parser.add_argument("--adv_factor", default=0.001, type=float)
    parser.add_argument("--early_stop", default=False,  type=bool)
    #parser.add_argument("--time", default=False,  type=bool)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--exp_factor", default=1, type=float)
    #parser.add_argument("--time", default=True, type=bool)
    #parser.add_argument("--data_size", default=2035, type=int)
    parser.add_argument("--mean", default=21.4, type=float)
    parser.add_argument("--std", default=17.3, type=float)
    parser.add_argument("--max", default=135, type=float)
    parser.add_argument("--double", default=False, type=bool)
    parser.add_argument("--triple", default=False, type=bool)
    parser.add_argument("--test", default=True, type=bool)
    parser.add_argument("--constraints", default='none')
    parser.add_argument("--mr", default=False, type=bool)
    parser.add_argument("--l2_reg", default=False, type=bool)
    parser.add_argument("--dim_channels", default=1, type=int)
    parser.add_argument("--time", default=True)
    parser.add_argument("--nn", default=True)
    #parser.add_argument("--test", default=True)
    parser.add_argument("--time_steps", type=int, default=1)
    parser.add_argument("--mass_violation", type=bool, default=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--time_sr", default=False)
    parser.add_argument("--constraints_window_size", default=4, type=int)
    parser.add_argument("--ensemble", default=False)
    
    return parser.parse_args()


def main(args):

    data = load_data(args)
    if args.double:
        model2 = models.ResNet2Up(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=4, noise=args.noise, constraints=args.constraints, dim=1, output_mr=args.mr)
        model1 = models.ResNet2(number_channels=args.number_channels, number_residual_blocks=args.number_residual_blocks, upsampling_factor=2, noise=args.noise, constraints=args.constraints)

        model1 = load_weights(model1, args.model_id)
        model2 = load_weights(model2, args.model_id2)
        
        add_string = args.model_id + '_and_' + args.model_id2+'_test'

        scores = evaluate_double_model(model1, model2, data, args, add_string)

        
        
    else:
        #model1 = load_model(args)

        #model1 = load_weights(model1, args.model_id)

        add_string = '_test'
        scores = evaluate_model( data, args, add_string)

        
        #add_string = args.model_id + '_evaluate_training' 
        
    #create_report(scores, args, add_string)
    main_scoring(args)
    
def load_weights(model, model_id):
    PATH = '/home/harder/constraint_generative_ml/models/'+model_id+'.pth'
    checkpoint = torch.load(PATH) # ie, model_best.pth.tar
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cuda')
    return model
    
if __name__ == '__main__':
    args = add_arguments()
    main(args)