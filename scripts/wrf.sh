#!/bin/sh

# none

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints none --batch_size 512 --model_id jmlr_wrf_none_1 

# gh

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints enforce_op --batch_size 512 --model_id jmlr_wrf_gh_1


# add

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints add --batch_size 512 --model_id jmlr_wrf_add_1


# mult

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints mult --batch_size 512 --model_id jmlr_wrf_mult_1

# softmax

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints softmax --batch_size 512 --model_id jmlr_wrf_softmax_1

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints none --batch_size 512 --model_id jmlr_wrf_none_2 

# gh

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints enforce_op --batch_size 512 --model_id jmlr_wrf_gh_2


# add

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints add --batch_size 512 --model_id jmlr_wrf_add_2 


# mult

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints mult --batch_size 512 --model_id jmlr_wrf_mult_2 

# softmax

python main.py --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints softmax --batch_size 512 --model_id jmlr_wrf_softmax_2 

#kronecker bicubic

#python scoring.py --dataset dataset38 --test_val_train test --model_id kronecker_wrf --model kronecker --factor 3
#python scoring.py --dataset dataset38 --test_val_train test --model_id bc_wrf --model bicubic --factor 3
#predict and evaluate QL
