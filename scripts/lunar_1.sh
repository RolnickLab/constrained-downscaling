#!/bin/sh



# softmax

python main.py --dataset dataset40 --scale minmax --epochs 100 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_lunar4_softmax_1 

# none

python main.py --dataset dataset40 --scale minmax --epochs 100 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_lunar4_none_2 

# gh

python main.py --dataset dataset40 --scale minmax --epochs 100 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_lunar4_gh_2


# add
# we are here currently 3am
python main.py --dataset dataset40 --scale minmax --epochs 100 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_lunar4_add_2 


# mult

python main.py --dataset dataset40 --scale minmax --epochs 100 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_lunar4_mult_2 

# softmax

python main.py --dataset dataset40 --scale minmax --epochs 100 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_lunar4_softmax_2

#kronecker bicubic

#python scoring.py --dataset dataset40 --test_val_train test --model_id kronecker_lunar4 --model kronecker --factor 4
#python scoring.py --dataset dataset40 --test_val_train test --model_id bc_lunar4 --model bicubic --factor 4
#predict and evaluate QL
