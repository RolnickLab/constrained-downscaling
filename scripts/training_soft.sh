#!/bin/sh

# none

python evaluate.py --test_val_train train --dataset dataset28 --scale minmax --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc4_soft_0
python scoring.py --dataset dataset28 --test_val_train train --model_id jmlr_wc4_soft_0 --model jmlr_wc4_soft_0 --factor 4 --nn True

python evaluate.py --test_val_train train --dataset dataset28 --scale minmax --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc4_soft_1
python scoring.py --dataset dataset28 --test_val_train train --model_id jmlr_wc4_soft_1 --model jmlr_wc4_soft_1 --factor 4 --nn True

python evaluate.py --test_val_train train --dataset dataset28 --scale minmax --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc4_soft_2
python scoring.py --dataset dataset28 --test_val_train train --model_id jmlr_wc4_soft_2 --model jmlr_wc4_soft_2 --factor 4 --nn True

# gh




