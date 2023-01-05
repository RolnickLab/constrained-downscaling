#!/bin/sh

# none

python main.py --dataset dataset38 --loss mass_constraints --alpha 0.99 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints none --batch_size 512 --model_id jmlr_wrf_soft_0

python main.py --dataset dataset38 --loss mass_constraints --alpha 0.99 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints none --batch_size 512 --model_id jmlr_wrf_soft_1 

# gh



python main.py --dataset dataset38 --loss mass_constraints --alpha 0.99 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints none --batch_size 512 --model_id jmlr_wrf_soft_2 
