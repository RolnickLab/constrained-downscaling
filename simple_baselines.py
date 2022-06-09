from skimage import transform
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torchgeometry as tgm
import argparse
from torch.utils.data import DataLoader, TensorDataset
import csv

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dataset26', help="choose a data set to use")
    parser.add_argument("--model", default='dataset23_resnet_sconstraints_st_0_256')
    parser.add_argument("--model_id", default='dataset23_resnet_sconstraints_st_0_256_fullprediction')
    parser.add_argument("--time", default=True)
    parser.add_argument("--nn", default=True)
    parser.add_argument("--time_steps", type=int, default=1)
    parser.add_argument("--mass_violation", type=bool, default=False)
    return parser.parse_args()

def main(args):
    #n = 24
    input_val = torch.load('./data/val/'+args.dataset+'/input_val.pt')
    target_val = torch.load('./data/val/'+args.dataset+'/target_val.pt')
    val_data = TensorDataset(input_val, target_val)
    pred = np.zeros(target_val.shape)
    print(pred.shape)
    factor = 16
    mse = 0
    mae = 0
    ssim = 0
    mass_violation = 0
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()
    ssim_criterion = tgm.losses.SSIM(window_size=11, max_val=torch.max(target_val) , reduction='mean')
    if args.nn:
        pred = torch.load('./data/prediction/'+args.model_id+'.pt')
        pred = pred.detach().cpu().numpy()
        print(pred.shape)
    for i,(lr, hr) in enumerate(val_data):
        im = lr.numpy()
        if args.time:
            #print(hr.shape)
            for j in range(args.time_steps):
                if args.model == 'bilinear':
                    pred[i,j,0,:,:] = np.array(Image.fromarray(im[j,0,...]).resize((4*lr.shape[2],4*lr.shape[2]), Image.BILINEAR))
                elif args.model == 'bicubic':
                    pred[i,j,0,:,:] = np.array(Image.fromarray(im[j,0,...]).resize((factor*lr.shape[2],factor*lr.shape[2]), Image.BICUBIC))
                elif args.model == 'bicubic_frame':
                    if j == 0:
                        pred[i,j,0,:,:] = np.array(Image.fromarray(im[0,0,...]).resize((4*lr.shape[2],4*lr.shape[2]), Image.BICUBIC))
                    elif j == 2:
                        pred[i,j,0,:,:] = np.array(Image.fromarray(im[1,0,...]).resize((4*lr.shape[2],4*lr.shape[2]), Image.BICUBIC))
                    else:
                        pred[i,j,0,:,:] =np.array(Image.fromarray(0.5*(im[1,0,...]+im[0,0,...])).resize((4*lr.shape[2],4*lr.shape[2]), Image.BICUBIC))
                elif args.model == 'kronecker':
                    pred[i,j,0,:,:] = np.kron(im[j,0,...], np.ones((4,4)))
                elif args.model=='frame_inter':
                    print(pred.shape, im.shape)
                    pred[i,j,0,... ] = 0.5*(im[0,0,...]+im[1,0,...])
                
                #print(l2_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item())
                mse += l2_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
                mae += l1_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
                ssim += ssim_criterion(torch.Tensor(pred[i,j:j+1,...]), hr[j:j+1,...]).item()
                if args.mass_violation:
                    mass_violation += np.mean( np.abs(transform.downscale_local_mean(pred[i,j,...], (1,8,8)) -im[j,...]))
                #print(l2_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item())
        elif args.model=='frame_inter':
            pred[i,... ] = 0.5*(im[0:1,...]+im[1:2,...])
            mse += l2_crit(torch.Tensor(pred[i,...]), hr).item()
            mae += l1_crit(torch.Tensor(pred[i,...]), hr).item()
            ssim += ssim_criterion(torch.Tensor(pred[i,...]), hr).item()
            #print(l1_crit(torch.Tensor(pred[i,...]), hr).item())
        else:
            if args.model == 'bilinear':
                pred[i,:,:] = np.array(Image.fromarray(im).resize((4*lr.shape[1],4*lr.shape[1]), Image.BILINEAR))
            elif args.model == 'bicubic':
                pred[i,:,:] = np.array(Image.fromarray(im).resize((4*lr.shape[1],4*lr.shape[1]), Image.BICUBIC))
        #elif args.model == 'kronecker':
            
            mse += l2_crit(torch.Tensor(pred[i,:,:]), hr).item()
            mae += l1_crit(torch.Tensor(pred[i,:,:]), hr).item()
            ssim += ssim_criterion(torch.Tensor(pred[i,:,:]).unsqueeze(0), hr.unsqueeze(0)).item()
            
    
    #torch.save(torch.Tensor(pred[:128,:,:]).unsqueeze(1), './data/prediction/'+args.dataset+'_'+args.model_id+'_prediction.pt')
    if args.time:
        print(input_val.shape[0])
        mse *= 1/(input_val.shape[0]*args.time_steps)
        mae *= 1/(input_val.shape[0]*args.time_steps)
        ssim *= 1/(input_val.shape[0]*args.time_steps)
        if args.mass_violation:
            mass_violation *= 1/(input_val.shape[0]*args.time_steps)
    else:
        mse *= 1/input_val.shape[0]   
        mae *= 1/input_val.shape[0] 
        ssim *= 1/input_val.shape[0] 
    psnr = calculate_pnsr(mse, torch.max(target_val)   )     
    scores = {'MSE':mse, 'RMSE':torch.sqrt(torch.Tensor([mse])), 'PSNR': psnr, 'MAE':mae, 'SSIM':1-ssim, 'Mass_violation': mass_violation}
    print(scores)
    create_report(scores, args)
    np.save('./data/prediction/bicubic_frame.npy', pred)
            
            
def calculate_pnsr(mse, max_val):
    return 20 * torch.log10(max_val / torch.sqrt(torch.Tensor([mse])))
                                            
def create_report(scores, args):
    args_dict = args_to_dict(args)
    #combine scorees and args dict
    args_scores_dict = args_dict | scores
    #save dict
    save_dict(args_scores_dict, args)
    
def args_to_dict(args):
    return vars(args)
    
                                            
def save_dict(dictionary, args):
    w = csv.writer(open('./data/score_log/'+args.model_id+'.csv', 'w'))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])
    
if __name__ == '__main__':
    args = add_arguments()
    main(args)