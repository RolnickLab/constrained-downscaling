#!/bin/sh

# none

python evaluate.py --test_val_train train --dataset dataset28 --scale minmax --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_wc4_none_2
python scoring.py --dataset dataset28 --test_val_train train --model_id jmlr_wc4_none_2 --model jmlr_wc4_none_2 --factor 4 --nn True

# gh

python evaluate.py --test_val_train train --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_wc4_gh_2

python scoring.py --dataset dataset28 --test_val_train train --model_id jmlr_wc4_gh_2 --model jmlr_wc4_gh_2 --factor 4 --nn True
# add

#we are here currently 3am
python evaluate.py --test_val_train train --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_wc4_add_2

python scoring.py --dataset dataset28 --test_val_train train --model_id jmlr_wc4_add_2 --model jmlr_wc4_add_2 --factor 4 --nn True
# mult

python evaluate.py --test_val_train train --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_wc4_mult_2 

python scoring.py --dataset dataset28 --test_val_train train --model_id jmlr_wc4_mult_2 --model jmlr_wc4_mult_2 --factor 4 --nn True
# softmax

python evaluate.py --test_val_train train --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_wc4_softmax_2 

#kronecker bicubic
python scoring.py --dataset dataset28 --test_val_train train --model_id jmlr_wc4_softmax_2 --model jmlr_wc4_softmax_2 --factor 4 --nn True


