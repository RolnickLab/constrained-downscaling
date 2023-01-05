#!/bin/sh

# none

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_ood_none_1 

# gh

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_ood_gh_1


# add

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_ood_add_1


# mult

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_ood_mult_1 

# softmax

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_ood_softmax_1

# none

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_ood_none_2 

# gh

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_ood_gh_2


# add

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_ood_add_2 


# mult

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_ood_mult_2 

# softmax

python main.py --dataset dataset34 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_ood_softmax_2 

#kronecker bicubic

#python scoring.py --dataset dataset34 --test_val_train test --model_id kronecker_ood --model kronecker --factor 4
#python scoring.py --dataset dataset34 --test_val_train test --model_id bc_ood --model bicubic --factor 4
#predict and evaluate QL
