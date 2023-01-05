#!/bin/sh

# none

python main.py --dataset dataset37 --scale minmax --epochs 200 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_noresm_soft_0 --loss mass_constraints --alpha 0.99

python main.py --dataset dataset37 --scale minmax --epochs 200 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_noresm_soft_1 --loss mass_constraints --alpha 0.99
python main.py --dataset dataset37 --scale minmax --epochs 200 --upsampling_factor 2 --factor 2 --constraints_window_size 2 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_noresm_soft_2 --loss mass_constraints --alpha 0.99

# gh

#

#kronecker bicubic

#python scoring.py --dataset dataset37 --test_val_train test --model_id kronecker_noresm --model kronecker --factor 2
#python scoring.py --dataset dataset37 --test_val_train test --model_id bc_noresm --model bicubic --factor 2
#predict and evaluate QL
