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
import properscoring as ps


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='dataset34', help="choose a data set to use")
    parser.add_argument("--model", default='resnet2_ood_softmax_test')
    parser.add_argument("--model_id", default='dataset34_resnet2_ood_softmax_test_test')
    #parser.add_argument("--time", default=True)
    parser.add_argument("--nn", default=False)
    parser.add_argument("--test_val_train", default='test')
    parser.add_argument("--time_steps", type=int, default=1)
    parser.add_argument("--mass_violation", type=bool, default=True)
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--time_sr", default=False)
    parser.add_argument("--ensemble", default=False)
    
    #args for model loading
    return parser.parse_args()

def main_scoring(args):
    #n = 24
    input_val = torch.load('./data/'+ args.test_val_train+'/'+args.dataset+'/input_'+ args.test_val_train+'.pt')
    target_val = torch.load('./data/'+ args.test_val_train+'/'+args.dataset+'/target_'+ args.test_val_train+'.pt')
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
    mean_abs_bias = 0
    mass_violation = 0
    ms_ssim = 0
    corr = 0
    crps = 0
    neg_mean = 0
    neg_num = 0
    
    l2_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()
    #ssim_criterion = StructuralSimilarityIndexMeasure() #tgm.losses.SSIM(window_size=11, max_val=max_val, reduction='mean')
    if args.nn:
        if args.ensemble:
            
            en_pred = torch.load('./data/prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'_ensemble.pt')
            

            pred = torch.mean(en_pred, dim=1)
            en_pred = en_pred.detach().cpu().numpy()
        else:
            
            pred = torch.load('./data/prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'.pt')
            
                
        pred = pred.detach().cpu().numpy()
        
        print(pred.shape)
    elif args.model == 'temp':
        pred = predict_T(args)
        pred = pred.detach().cpu().numpy()
    elif args.model == 'ql':
        pred = predict_ql(args)
        pred = pred.detach().cpu().numpy()
    for i,(lr, hr) in enumerate(val_data):
        im = lr.numpy()
        
        #print(hr.shape)
        for j in range(args.time_steps):
            if args.model == 'bilinear':
                pred[i,j,0,:,:] = np.array(Image.fromarray(im[j,0,...]).resize((factor*lr.shape[2],factor*lr.shape[2]), Image.BILINEAR))
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
                pred[i,j,0,:,:] = np.kron(im[j,0,...], np.ones((args.factor,args.factor)))
            elif args.model=='frame_inter':
                print(pred.shape, im.shape)
                pred[i,j,0,... ] = 0.5*(im[0,0,...]+im[1,0,...])
            
            '''
            mse_loss = l2_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()

            print(i, mse_loss)'''
            mse += l2_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
            mae += l1_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item()
            mean_bias += torch.mean( hr[j,...]-torch.Tensor(pred[i,j,...]))
            mean_abs_bias += torch.abs(torch.mean( hr[j,...]-torch.Tensor(pred[i,j,...])))
            corr += pearsonr(torch.Tensor(pred[i,j,...]).flatten(),  hr[j,...].flatten())
            ms_ssim += multiscale_structural_similarity_index_measure(torch.Tensor(pred[i,j:j+1,...]), hr[j:j+1,...], data_range=max_val-min_val, kernel_size=11, betas=(0.2856, 0.3001, 0.2363))#0.0448, 0.2856, 0.3001, 0.2363))
            ssim += structural_similarity_index_measure(torch.Tensor(pred[i,j:j+1,...]), hr[j:j+1,...] , data_range=max_val-min_val, kernel_size=11)#ssim_criterion(torch.Tensor(pred[i,j:j+1,...]), hr[j:j+1,...]).item()
            neg_num += np.sum(pred[i,j,...] < 0)
            neg_mean += np.sum(pred[pred < 0])/(pred.shape[-1]*pred.shape[-1])
            if args.ensemble:
                crps_ens = crps_ensemble(hr[j,0,...].numpy(), en_pred[i,:,j,0,...])
                crps += crps_ens
            if args.mass_violation:
                if args.time_sr:
                    if j==0:
                        mass_violation += np.mean( np.abs(transform.downscale_local_mean(pred[i,j,...], (1,args.factor,args.factor)) -im[0,...]))
                    elif j==2:
                        mass_violation += np.mean( np.abs(transform.downscale_local_mean(pred[i,j,...], (1,args.factor,args.factor)) -im[1,...]))
                else:
                    mass_violation += np.mean( np.abs(transform.downscale_local_mean(pred[i,j,...], (1,args.factor,args.factor)) -im[j,...]))
            #print(l2_crit(torch.Tensor(pred[i,j,...]), hr[j,...]).item())
        if args.model=='frame_inter':
            pred[i,... ] = 0.5*(im[0:1,...]+im[1:2,...])
            mse += l2_crit(torch.Tensor(pred[i,...]), hr).item()
            mae += l1_crit(torch.Tensor(pred[i,...]), hr).item()
            ssim += ssim_criterion(torch.Tensor(pred[i,...]), hr).item()
            #print(l1_crit(torch.Tensor(pred[i,...]), hr).item())
        '''
        else:
            if args.model == 'bilinear':
                pred[i,:,:] = np.array(Image.fromarray(im).resize((4*lr.shape[1],4*lr.shape[1]), Image.BILINEAR))
            elif args.model == 'bicubic':
                pred[i,:,:] = np.array(Image.fromarray(im).resize((4*lr.shape[1],4*lr.shape[1]), Image.BICUBIC))
        #elif args.model == 'kronecker':
            
            mse += l2_crit(torch.Tensor(pred[i,:,:]), hr).item()
            mae += l1_crit(torch.Tensor(pred[i,:,:]), hr).item()
            ssim += ssim_criterion(torch.Tensor(pred[i,:,:]).unsqueeze(0), hr.unsqueeze(0)).item()'''
            
    if args.model == 'bicubic' or args.model == 'kronecker' or args.model == 'frame':
        torch.save(torch.Tensor(pred), './data/prediction/'+args.dataset+'_'+args.model_id+ '_' + args.test_val_train+'.pt')
    #torch.save(torch.Tensor(pred[:128,:,:]).unsqueeze(1), './data/prediction/'+args.dataset+'_'+args.model_id+'_prediction.pt')
    
    print(input_val.shape[0])
    mse *= 1/(input_val.shape[0]*args.time_steps)
    mae *= 1/(input_val.shape[0]*args.time_steps)
    ssim *= 1/(input_val.shape[0]*args.time_steps)
    mean_bias *= 1/(input_val.shape[0]*args.time_steps)
    mean_abs_bias *= 1/(input_val.shape[0]*args.time_steps)
    corr *= 1/(input_val.shape[0]*args.time_steps)
    ms_ssim *= 1/(input_val.shape[0]*args.time_steps)
    crps *=1/(input_val.shape[0]*args.time_steps)
    neg_mean *=1/(input_val.shape[0]*args.time_steps)
    #neg_num *=1/(input_val.shape[0]*args.time_steps)
    if args.mass_violation:
        if args.model == 'bicubic_frame':
            mass_violation *= 1/(input_val.shape[0]*2)
        else:
            mass_violation *= 1/(input_val.shape[0]*args.time_steps)
        
    '''else:
        mse *= 1/input_val.shape[0]   
        mae *= 1/input_val.shape[0] 
        ssim *= 1/input_val.shape[0] '''
    psnr = calculate_pnsr(mse, target_val.max() )   
    rmse = torch.sqrt(torch.Tensor([mse])).numpy()[0]
    ssim = float(ssim.numpy())
    ms_ssim =float( ms_ssim.numpy())
    psnr = psnr.numpy()
    corr = float(corr.numpy())
    mean_bias = float(mean_bias.numpy())
    mean_abs_bias = float(mean_abs_bias.numpy())
    scores = {'MSE':mse, 'RMSE':rmse, 'PSNR': psnr[0], 'MAE':mae, 'SSIM':ssim,  'MS SSIM': ms_ssim, 'Pearson corr': corr, 'Mean bias': mean_bias, 'Mean abs bias': mean_abs_bias, 'Mass_violation': mass_violation, 'neg mean': neg_mean, 'neg num': neg_num,'CRPS': crps}
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

'''
def crps_ensemble(target, prediction):
    #print(target.shape, prediction.shape)
    crps_sum = 0
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            crps_sum += ps.crps_ensemble(target[i,j], prediction[:,i,j])
    
    return crps_sum/(target.shape[0]*target.shape[1])'''


'''
def crps_ensemble(observation, forecasts):
    fc = forecasts.copy()
    fc.sort(axis=-1)
    obs = observation
    fc_below = fc<obs[...,None]
    crps = np.zeros_like(obs)
    print(fc.shape, obs.shape, crps.shape)
    for i in range(fc.shape[-1]):
        below = fc_below[...,i]
        weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
        crps[below] += weight * (obs[below]-fc[...,i][below])

    for i in range(fc.shape[-1]-1,-1,-1):
        above  = ~fc_below[...,i]
        k = fc.shape[-1]-1-i
        weight = ((k+1)**2 - k**2) / fc.shape[-1]**2
        crps[above] += weight * (fc[...,i][above]-obs[above])

    return np.mean(crps)'''

def crps_ensemble(observation, forecasts):
    fc = forecasts.copy()
    fc.sort(axis=0)
    obs = observation
    fc_below = fc<obs[None,...]
    crps = np.zeros_like(obs)
    #print(fc.shape, obs.shape, crps.shape)
    for i in range(fc.shape[0]):
        below = fc_below[i,...]
        weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
        crps[below] += weight * (obs[below]-fc[i,...][below])

    for i in range(fc.shape[0]-1,-1,-1):
        above  = ~fc_below[i,...]
        k = fc.shape[0]-1-i
        weight = ((k+1)**2 - k**2) / fc.shape[0]**2
        crps[above] += weight * (fc[i,...][above]-obs[above])

    return np.mean(crps)
                                            
def create_report(scores, args):
    args_dict = args_to_dict(args)
    #combine scorees and args dict
    args_scores_dict = args_dict | scores
    #save dict
    save_dict(args_scores_dict, args)
    
def args_to_dict(args):
    return vars(args)
    
                                            
def save_dict(dictionary, args):
    if args.model=='temp':
        fn = './data/score_log/jmlr_t4_'+args.model_id+ '_' + args.test_val_train+'.csv'
    elif args.model =='ql':
        fn = './data/score_log/jmlr_ql4_'+args.model_id+ '_' + args.test_val_train+'.csv'
    else:
        fn = './data/score_log/'+args.model_id+ '_' + args.test_val_train+'.csv'
    w = csv.writer(open(fn, 'w'))
    # loop over dictionary keys and values
    for key, val in dictionary.items():
        # write every key and value to file
        w.writerow([key, val])
        
        
def predict_T(args):
    #T = p7(qv*rho)
    #load Qv
    qv = torch.load('./data/prediction/dataset41'+'_jmlr_qv4_'+args.model_id+ '_' + args.test_val_train+'.pt')
    ql = torch.load('./data/prediction/dataset44'+'_jmlr_ql4_'+args.model_id+ '_' + args.test_val_train+'.pt')
    #lv = 2.5008*1e6+(1846.1-4218)*(hr_t-273.16)
    s = torch.load('./data/prediction/dataset43'+'_jmlr_s4_'+args.model_id+ '_' + args.test_val_train+'.pt')
    
    pred = (s-2.5008*1e6*qv+(1846.1-4218)*273.16*qv)/((1-qv)*1004.7+ql*1846.1+(1846.1-4218)*qv)
    #save T
    torch.save(pred, './data/prediction/dataset45'+'_jmlr_t4_'+args.model_id+ '_' + args.test_val_train+'.pt')
    return pred

def predict_ql(args):
    #ql = (s-lv*qv-(1-qv)*1004.7*hr_t)/(hr_t*1846.1)
    #load T
    hr_t = torch.load('./data/prediction/dataset45'+'_jmlr_t4_'+args.model_id+ '_' + args.test_val_train+'.pt')
    lv = 2.5008*1e6+(1846.1-4218)*(hr_t-273.16)
    #load qv
    qv = torch.load('./data/prediction/dataset42'+'_jmlr_rho4_'+args.model_id+ '_' + args.test_val_train+'.pt')
    #load s
    s = torch.load('./data/prediction/dataset43'+'_jmlr_s4_'+args.model_id+ '_' + args.test_val_train+'.pt')
    pred = (s-lv*qv-(1-qv)*1004.7*hr_t)/(hr_t*1846.1)
    torch.save(pred, './data/prediction/dataset44'+'_jmlr_ql4_'+args.model_id+ '_' + args.test_val_train+'.pt')
    return pred

    
if __name__ == '__main__':
    args = add_arguments()
    main_scoring(args)