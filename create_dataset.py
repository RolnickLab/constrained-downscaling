from skimage import transform
import torch
import numpy as np
import netCDF4 as nc
import numpy
from torch.utils.data import TensorDataset
import os
from numpy.random import default_rng

def main():
    #load nc file
    fn = '/home/harder/constraint_generative_ml/data/adaptor.mars.internal-1646234417.8891354-21152-12-ccec0740-cdb2-4e9d-a89b-4545158e13c1.nc'
    ds = nc.Dataset(fn)
    #
    dataset_name = 'dataset23'
    train_val_test = 'all'
    
    is_time = False
    time_sr = False
    is_deep_voxel = False
    time_steps = 1
    time_steps_in = 1
    hr_size = 128
    lr_size = 32
    mr_size = 64
    factor = 4
    mid = False
    water_content = True
    lon = 1440
    lat = 721
    start_time = 0#545+109#545+109# 910+182
    time =  600#61368
    n_lons = int(lon/hr_size)
    n_lats = int(lat/hr_size)
    channels = 1
    if is_time or time_sr or is_deep_voxel:
        n_times = int(time/time_steps)
    else:
        n_times = time
    n = int(n_lons*n_lats*(n_times+1))
    n_train = 20000#int(0.8*n)
    n_val = 4000#int(0.1*n)
    print(n_lons, n_lats, n_times, n)
    
    hr_data = np.zeros((n,time_steps,channels,hr_size,hr_size))
    lr_data = np.zeros((n,time_steps_in,channels,lr_size,lr_size))
    if mid:
        mr_data = np.zeros((n,time_steps,channels,mr_size,mr_size))
    
    count = 0
    print(n)
    '''
    for j in range(n_lons):
        for k in range(n_lats):
            for i in np.arange(start_time,time+start_time,time_steps):
                hr = ds['tcw'][i:i+time_steps,hr_size*k:hr_size*(k+1),hr_size*j:hr_size*(j+1)]
                lr = transform.downscale_local_mean(hr, (1,factor,factor))
                hr_data[count,0,0,:,:] = lr[0,1,...]
                lr_data[count,0,0,:,:] = lr[0,0,...]
                lr_data[count,1,0,:,:] = lr[0,2,...]
                count += 1'''

    

    rng = default_rng()
    time_inds = rng.choice(61368, size=600, replace=False)

    for j in range(n_lons):
        for k in range(n_lats):
            for i in np.arange(0,time,time_steps):
                l = time_inds[i]
                hr_data[count,:,0,:,:] = ds['tcw'][l:l+time_steps,hr_size*k:hr_size*(k+1),hr_size*j:hr_size*(j+1)]
                lr_data[count,:,0,:,:] = transform.downscale_local_mean(hr_data[count,:,0,:,:], (1,factor,factor))
                count += 1
    
    if not os.path.exists('./train/'+dataset_name):
        os.makedirs('./train/'+dataset_name)
        os.makedirs('./val/'+dataset_name)
        os.makedirs('./test/'+dataset_name)
        
    if not is_time:
        rng_state = numpy.random.get_state()
        numpy.random.permutation(lr_data)
        numpy.random.set_state(rng_state)
        numpy.random.permutation(hr_data)
        
    
    if train_val_test == 'all':
        torch.save(torch.Tensor(lr_data[:n_train,...]) ,'./train/'+dataset_name+'/input_train.pt')
        torch.save(torch.Tensor(hr_data[:n_train,...]) ,'./train/'+dataset_name+'/target_train.pt')
        torch.save(torch.Tensor(lr_data[n_train:n_train+n_val,...]) ,'./val/'+dataset_name+'/input_val.pt') 
        torch.save(torch.Tensor(hr_data[n_train:n_train+n_val,...]) ,'./val/'+dataset_name+'/target_val.pt')
        torch.save(torch.Tensor(lr_data[n_train+n_val:n_train+2*n_val,...]), './test/'+dataset_name+'/input_test.pt')
        torch.save(torch.Tensor(hr_data[n_train+n_val:n_train+2*n_val,...]),'./test/'+dataset_name+'/target_test.pt')
    elif train_val_test == 'train':
        inds = [0,2]
        torch.save(torch.Tensor(lr_data[:,inds,...] ),'./train/'+dataset_name+'/input_train.pt')
        torch.save(torch.Tensor(lr_data[:,1:2,...]) ,'./train/'+dataset_name+'/target_train.pt')
    elif train_val_test == 'val':
        inds = [0,2]
        torch.save(torch.Tensor(lr_data[:,inds,...] ),'./val/'+dataset_name+'/input_val.pt')
        torch.save(torch.Tensor(lr_data[:,1:2,...]) ,'./val/'+dataset_name+'/target_val.pt')
    elif train_val_test == 'test':
        torch.save(torch.Tensor(lr_data), './test/'+dataset_name+'/input_test.pt')
        torch.save(torch.Tensor(hr_data),'./test/'+dataset_name+'/target_test.pt')
            
    
    
    
    '''
    if is_time:
        if water_content:
            for j in range(n_lons):
                for k in range(n_lats):
                    for i in np.arange(0,time,time_steps):
                        hr_data[count,:,0,:,:] = ds['tcw'][i:i+time_steps,hr_size*k:hr_size*(k+1),hr_size*j:hr_size*(j+1)]
                        lr_data[count,:,0,:,:] = transform.downscale_local_mean(hr_data[count,:,0,:,:], (1,factor,factor))
                        lr
                        count += 1
                        
    elif time_sr:
        for j in range(n_lons):
                for k in range(n_lats):
                    for i in np.arange(start_time,time+start_time,time_steps):
                        hr_data[count,:,0,:,:] = ds['tcw'][i:i+time_steps,hr_size*k:hr_size*(k+1),hr_size*j:hr_size*(j+1)]
                        spatiallr_data[count,:,0,:,:] = transform.downscale_local_mean(hr_data[count,:,0,:,:], (1,factor,factor))
                        for l in range(time_steps):
                            if i==0:
                                templr_data[count,l,0,:,:] = spatiallr_data[count,l,0,:,:]
                            elif l % 2 == 0:
                                templr_data[count,l,0,:,:] = spatiallr_data[count,l,0,:,:]
                                templr_data[count,l-1,0,:,:] = 0.5*(spatiallr_data[count,l,0,:,:]+spatiallr_data[count,l-2,0,:,:])
                        count += 1
    elif is_deep_voxel:
        for j in range(n_lons):
                for k in range(n_lats):
                    for i in np.arange(start_time,time+start_time,time_steps):
                        hr_data[count,:,0,:,:] = ds['tcw'][i:i+time_steps,hr_size*k:hr_size*(k+1),hr_size*j:hr_size*(j+1)]
                        lr_data[count,:,0,:,:] = transform.downscale_local_mean(hr_data[count,:,0,:,:], (1,factor,factor))
                        templr_data[count,0,:,:] = lr_data[count,0,0,:,:]
                        templr_data[count,1,:,:] = lr_data[count,2,0,:,:]
                        spatiallr_data[count,0,:,:] = lr_data[count,1,0,:,:]
                        count += 1
        
    else:
        for i in range(192):
            for j in range(11):
                for k in range(5):
                    hr_q = ds['q'][i,hr_size*k:hr_size*(k+1),hr_size*j:hr_size*(j+1)]
                    hr_t = ds['t'][i,hr_size*k:hr_size*(k+1),hr_size*j:hr_size*(j+1)]
                    hr_r = 850/(hr_t*hr_q)
                    #hr_ql = ds['clwc'][i,hr_size*k:hr_size*(k+1),hr_size*j:hr_size*(j+1)]
                    #hr_lv = 2.5008*1e6+(1846.1-4218)*(hr_t-273.16)
                    #hr_s = ((1-hr_q)*1004.7+hr_ql*1846.1)*hr_t+hr_lv*hr_q
                    hr_data[count,0,:,:] = hr_r
                    #hr_data[count,1,:,:] = hr_r
                    #hr_data[count,2,:,:] = hr_s
                    lr_q = transform.downscale_local_mean(hr_q, (factor, factor))
                    lr_r = transform.downscale_local_mean(hr_r ,(factor, factor))
                    #lr_s = transform.downscale_local_mean(hr_s ,(factor, factor))
                    lr_data[count,0,:,:] = lr_r
                    #lr_data[count,1,:,:] = lr_r
                    #lr_data[count,2,:,:] = lr_s
                    mr_r = transform.downscale_local_mean(hr_r ,(2, 2))
                    mr_data[count,0,:,:] = mr_r
                    count +=1            
    print(count)         
    
    
    if time_sr:
        rng_state = numpy.random.get_state()
        numpy.random.permutation(spatiallr_data)
        numpy.random.set_state(rng_state)
        numpy.random.permutation(hr_data)
        numpy.random.set_state(rng_state)
        numpy.random.permutation(templr_data)
    else:
        rng_state = numpy.random.get_state()
        numpy.random.permutation(lr_data)
        numpy.random.set_state(rng_state)
        numpy.random.permutation(hr_data)
    
    if not os.path.exists('./train/'+dataset_name):
        os.makedirs('./train/'+dataset_name)
        os.makedirs('./val/'+dataset_name)
        os.makedirs('./test/'+dataset_name)
        
    if time_sr or is_deep_voxel:
        if train_val_test == 'all':
            torch.save(torch.Tensor(templr_data[:n_train,...]) ,'./train/'+dataset_name+'/input_train.pt')
            torch.save(torch.Tensor(spatiallr_data[:n_train,...]) ,'./train/'+dataset_name+'/mid_train.pt')
            torch.save(torch.Tensor(hr_data[:n_train,...]) ,'./train/'+dataset_name+'/target_train.pt')
            torch.save(torch.Tensor(templr_data[n_train:n_train+n_val,...]) ,'./val/'+dataset_name+'/input_val.pt') #n_train:n_train+n_val
            torch.save(torch.Tensor(spatiallr_data[n_train:n_train+n_val,...]) ,'./val/'+dataset_name+'/mid_val.pt') #[n_train:n_train+n_val,...]
            torch.save(torch.Tensor(hr_data[n_train:n_train+n_val,...]) ,'./val/'+dataset_name+'/target_val.pt')
            torch.save(torch.Tensor(templr_data[n_train+n_val:,...]), './test/'+dataset_name+'/input_test.pt')
            torch.save(torch.Tensor(spatiallr_data[n_train+n_val:,...]), './test/'+dataset_name+'/mid_test.pt')
            torch.save(torch.Tensor(hr_data[n_train+n_val:,...]),'./test/'+dataset_name+'/target_test.pt')
        elif train_val_test == 'train':
            torch.save(torch.Tensor(templr_data) ,'./train/'+dataset_name+'/input_train.pt')
            torch.save(torch.Tensor(spatiallr_data) ,'./train/'+dataset_name+'/mid_train.pt')
            torch.save(torch.Tensor(hr_data) ,'./train/'+dataset_name+'/target_train.pt')
        elif train_val_test == 'val':
            torch.save(torch.Tensor(templr_data) ,'./val/'+dataset_name+'/input_val.pt') #n_train:n_train+n_val
            torch.save(torch.Tensor(spatiallr_data) ,'./val/'+dataset_name+'/mid_val.pt') #[n_train:n_train+n_val,...]
            torch.save(torch.Tensor(hr_data) ,'./val/'+dataset_name+'/target_val.pt')
        elif train_val_test == 'test':
            torch.save(torch.Tensor(templr_data), './test/'+dataset_name+'/input_test.pt')
            torch.save(torch.Tensor(spatiallr_data), './test/'+dataset_name+'/mid_test.pt')
            torch.save(torch.Tensor(hr_data),'./test/'+dataset_name+'/target_test.pt')
            
    else:
        torch.save(torch.Tensor(lr_data[:n_train,...]) ,'./train/'+dataset_name+'/input_train.pt')
        torch.save(torch.Tensor(hr_data[:n_train,...]) ,'./train/'+dataset_name+'/target_train.pt')
        torch.save(torch.Tensor(lr_data[n_train:n_train+n_val,...]) ,'./val/'+dataset_name+'/input_val.pt') 
        torch.save(torch.Tensor(hr_data[n_train:n_train+n_val,...]) ,'./val/'+dataset_name+'/target_val.pt')
        torch.save(torch.Tensor(lr_data[n_train+n_val:,...]), './test/'+dataset_name+'/input_test.pt')
        torch.save(torch.Tensor(hr_data[n_train+n_val:,...]),'./test/'+dataset_name+'/target_test.pt')
        if mid:
            torch.save(torch.Tensor(mr_data[:n_train,...]) ,'./train/'+dataset_name+'/mid_train.pt')
            torch.save(torch.Tensor(mr_data[n_train:n_train+n_val,...]) ,'./val/'+dataset_name+'/mid_val.pt')
            torch.save(torch.Tensor(mr_data[n_train+n_val:,...]), './test/'+dataset_name+'/mid_test.pt')
            '''

if __name__ == '__main__':
    main()
