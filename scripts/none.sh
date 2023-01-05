#!/bin/sh

# predict rho

# predict rho
python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_wc16_none_0

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_wc16_none_1

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_wc16_none_2

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_wc16_add_0

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_wc16_add_1

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_wc16_add_2

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_wc16_gh_0

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_wc16_gh_1

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_wc16_gh_2

python scoring.py --dataset dataset31 --test_val_train test --model_id bicubic_wc16 --model bicubic --factor 16
