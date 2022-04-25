import torch
import argparse
from training import evaluate_double_model, create_report
import models
from utils import load_data

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dataset15', help="choose a data set to use")
    parser.add_argument("--scale", default='standard', help="standard, minmax, none")
    parser.add_argument("--model", default='gan')
    parser.add_argument("--model_id", default='deep_voxel_flow')
    parser.add_argument("--model_id2", default='conv_gru_3_constraints')
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
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--dim", default=1, type=int)
    parser.add_argument("--dim_out", default=1, type=int)
    parser.add_argument("--exp_factor", default=1, type=float)
    parser.add_argument("--time", default=True, type=bool)
    parser.add_argument("--nsteps_out", default=3, type=int)
    parser.add_argument("--nsteps_in", default=2, type=int)
    parser.add_argument("--norm_dim", default=1, type=int)
    return parser.parse_args()


def main(args):

    data = load_data(args)
    
    model1 = models.VoxelFlow()
    model2 = models.ConvGRUGeneratorDet()
    
    model1 = load_weights(model1, args.model_id)
    model2 = load_weights(model2, args.model_id2)
     
    scores = evaluate_double_model(model1, model2, data, args)
    
    evaluate_double_model(model1, model2, data, args)
    add_string = args.model_id + '_and_' + args.model_id2
    create_report(scores, args, add_string)
    
def load_weights(model, model_id):
    PATH = '/home/harder/constraint_generative_ml/models/'+model_id+'.pth'
    checkpoint = torch.load(PATH) # ie, model_best.pth.tar
    model.load_state_dict(checkpoint['state_dict'])
    model.to('cuda')
    return model
    
if __name__ == '__main__':
    args = add_arguments()
    main(args)