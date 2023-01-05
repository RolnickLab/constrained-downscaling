#!/bin/sh

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_lunar2_none_0

# gh

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_lunar2_gh_0


# add

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_lunar2_add_0 


# mult

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_lunar2_mult_0 

# softmax

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_lunar2_softmax_0

# none
# none

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_lunar2_none_1 

# gh

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_lunar2_gh_1


# add

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_lunar2_add_1 


# mult

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_lunar2_mult_1 

# softmax

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_lunar2_softmax_1 

# none

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_lunar2_none_2 

# gh

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_lunar2_gh_2


# add

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_lunar2_add_2 


# mult

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_lunar2_mult_2 

# softmax

python main.py --dataset dataset39 --scale minmax --epochs 100 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_lunar2_softmax_2

#kronecker bicubic

#python scoring.py --dataset dataset40 --test_val_train test --model_id kronecker_lunar4 --model kronecker --factor 4
#python scoring.py --dataset dataset40 --test_val_train test --model_id bc_lunar4 --model bicubic --factor 4
#predict and evaluate QL
