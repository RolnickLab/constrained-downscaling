from skimage import transform
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torchgeometry as tgm
import argparse
from torch.utils.data import DataLoader, TensorDataset
import csv
from torchmetrics.functional import multiscale_structural_similarity_index_measure, structural_similarity_index_measure


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dataset34', help="choose a data set to use")
    parser.add_argument("--model", default='resnet2_ood_softmax_test')
    parser.add_argument("--model_id", default='dataset34_resnet2_ood_softmax_test_test')
    parser.add_argument("--time", default=True)
    parser.add_argument("--nn", default=True)
    parser.add_argument("--test", default=True)
    parser.add_argument("--time_steps", type=int, default=1)
    parser.add_argument("--mass_violation", type=bool, default=True)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--time_sr", default=False)
    
    #args for model loading
    return parser.parse_args()

def main_scoring(args):
    #n = 24
    input_val = torch.load('./data/test/'+args.dataset+'/input_test.pt')
    target_val = torch.load('./data/test/'+args.dataset+'/target_test.pt')
    #target_val = torch.load('./data/test/'+args.dataset+'/target_test.pt')
    #target_val = torch.load('./data/test/dataset28/target_test.pt')
    val_data = TensorDataset(input_val, target_val)
    pred = np.zeros(target_val.shape)
    max_val = target_val.max()
    min_val = target_val.min()
    print(pred.shape)
    factor = args.factor
    mse = 0
    mae = 0
    ssim = 0
    mean_bias = 0
    mass_violation = 0
    ms_ssim = 0
    corr = 0
    crps = 0
    
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()
    #ssim_criterion = StructuralSimilarityIndexMeasure() #tgm.losses.SSIM(window_size=11, max_val=max_val, reduction='mean')
    if args.nn:
        if args.ensemble:
            en_pred = torch.load('./data/prediction/'+args.dataset+'_'+args.model_id+'_test_ensemble.pt')
            pred = torch.mean(en_pred, dim=1)
        else:
            pred = torch.load('./data/prediction/'+args.dataset+'_'+args.model_id+'_test.pt')
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
                '''
                mse_loss = l2_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
                
                print(i, mse_loss)'''
                mse += l2_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
                mae += l1_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
                mean_bias += torch.mean( hr[j,...]-torch.abs(torch.Tensor(pred[i,j,...])))
                corr += pearsonr(torch.Tensor(pred[i,j,...]).flatten(),  hr[j,...].flatten())
                ms_ssim += multiscale_structural_similarity_index_measure(torch.Tensor(pred[i,j:j+1,...]), hr[j:j+1,...], data_range=max_val-min_val, kernel_size=11, betas=(0.2856, 0.3001, 0.2363))#0.0448, 0.2856, 0.3001, 0.2363))
                ssim += structural_similarity_index_measure(torch.Tensor(pred[i,j:j+1,...]), hr[j:j+1,...] , data_range=max_val-min_val, kernel_size=11)#ssim_criterion(torch.Tensor(pred[i,j:j+1,...]), hr[j:j+1,...]).item()
                if args.ensemble:
                    crps_ens = crps_ensemble(hr[j,0,0,...].numpy(), np.swapaxis(np.swapaxis(pred[i,:,j,0,0,...], 0,1),1,2))
                    crps += np.mean(crps_ens)
                if args.mass_violation:
                    if args.time_sr:
                        if j==0:
                            mass_violation += np.mean( np.abs(transform.downscale_local_mean(pred[i,j,...], (1,args.factor,args.factor)) -im[0,...]))
                        elif j==2:
                            mass_violation += np.mean( np.abs(transform.downscale_local_mean(pred[i,j,...], (1,args.factor,args.factor)) -im[1,...]))
                    else:
                        mass_violation += np.mean( np.abs(transform.downscale_local_mean(pred[i,j,...], (1,args.factor,args.factor)) -im[j,...]))
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
        mean_bias *= 1/(input_val.shape[0]*args.time_steps)
        corr *= 1/(input_val.shape[0]*args.time_steps)
        ms_ssim *= 1/(input_val.shape[0]*args.time_steps)
        
        if args.mass_violation:
            if args.time_sr:
                mass_violation *= 1/(input_val.shape[0]*args.time_steps)
            else:
                mass_violation *= 1/(input_val.shape[0]) #what is the 2 doing here?
    else:
        mse *= 1/input_val.shape[0]   
        mae *= 1/input_val.shape[0] 
        ssim *= 1/input_val.shape[0] 
    psnr = calculate_pnsr(mse, target_val.max() )     
    scores = {'MSE':mse, 'RMSE':torch.sqrt(torch.Tensor([mse])), 'PSNR': psnr, 'MAE':mae, 'SSIM':ssim, 'Mass_violation': mass_violation, 'Mean bias': mean_bias, 'MS SSIM': ms_ssim, 'Pearson corr': corr, 'CRPS': crps}
    print(scores)
    create_report(scores, args)
    #np.save('./data/prediction/bic.npy', pred)
            
            
def calculate_pnsr(mse, max_val):
    return 20 * torch.log10(max_val / torch.sqrt(torch.Tensor([mse])))
                    
def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`
    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor
    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def crps_ensemble(observation, forecasts):
    fc = forecasts.copy()
    fc.sort(axis=-1)
    obs = observation
    fc_below = fc<obs[...,None]
    crps = np.zeros_like(obs)

    for i in range(fc.shape[-1]):
        below = fc_below[...,i]
        weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
        crps[below] += weight * (obs[below]-fc[...,i][below])

    for i in range(fc.shape[-1]-1,-1,-1):
        above  = ~fc_below[...,i]
        k = fc.shape[-1]-1-i
        weight = ((k+1)**2 - k**2) / fc.shape[-1]**2
        crps[above] += weight * (fc[...,i][above]-obs[above])

    return crps
                                            
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
    main_scoring(args)