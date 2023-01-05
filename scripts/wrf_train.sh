#!/bin/sh

# none

python evaluate.py --test_val_train train --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints none --batch_size 512 --model_id jmlr_wrf_none_0 
python scoring.py --dataset dataset38 --test_val_train train --model_id jmlr_wrf_none_0 --model jmlr_wrf_none_0 --factor 4 --nn True
# gh

python evaluate.py --test_val_train train --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints soft --batch_size 512 --model_id jmlr_wrf_soft_0
python scoring.py --dataset dataset38 --test_val_train train --model_id jmlr_wrf_soft_0 --model jmlr_wrf_soft_0 --factor 4 --nn True

python evaluate.py --test_val_train train --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints enforce_op --batch_size 512 --model_id jmlr_wrf_gh_0
python scoring.py --dataset dataset38 --test_val_train train --model_id jmlr_wrf_gh_0 --model jmlr_wrf_gh_0 --factor 4 --nn True

# add

python evaluate.py --test_val_train train --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints add --batch_size 512 --model_id jmlr_wrf_add_0 
python scoring.py --dataset dataset38 --test_val_train train --model_id jmlr_wrf_add_0 --model jmlr_wrf_add_0 --factor 4 --nn True

# mult

python evaluate.py --test_val_train train --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints mult --batch_size 512 --model_id jmlr_wrf_mult_0 
python scoring.py --dataset dataset38 --test_val_train train --model_id jmlr_wrf_mult_0 --model jmlr_wrf_mult_0 --factor 4 --nn True
# softmax

python evaluate.py --test_val_train train --dataset dataset38 --scale minmax --epochs 200 --upsampling_factor 3 --factor 3 --constraints_window_size 3 --model resnet3 --constraints softmax --batch_size 512 --model_id jmlr_wrf_softmax_0 
python scoring.py --dataset dataset38 --test_val_train train --model_id jmlr_wrf_softmax_0 --model jmlr_wrf_softmax_0 --factor 4 --nn True
#kronecker bicubic

python scoring.py --test_val_train train --dataset dataset38 --test_val_train train --model_id kronecker_wrftrain --model kronecker --factor 3
python scoring.py --test_val_train train --dataset dataset38 --test_val_train train --model_id bc_wrftrain --model bicubic --factor 3
#predict and evaluate QL
