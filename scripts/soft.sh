#!/bin/sh

# 4x
python main.py --dataset dataset28 --loss mass_constraints --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc4_soft_0 --alpha 0.99

python main.py --dataset dataset28 --loss mass_constraints --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc4_soft_1 --alpha 0.99
python main.py --dataset dataset28 --loss mass_constraints --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc4_soft_2 --alpha 0.99

# 8x
python main.py --dataset dataset30 --loss mass_constraints --scale minmax --epochs 200 --upsampling_factor 8 --factor 8 --constraints_window_size 8 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc8_soft_0 --alpha 0.99
python main.py --dataset dataset30 --loss mass_constraints --scale minmax --epochs 200 --upsampling_factor 8 --factor 8 --constraints_window_size 8 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc8_soft_1 --alpha 0.99
python main.py --dataset dataset30 --loss mass_constraints --scale minmax --epochs 200 --upsampling_factor 8 --factor 8 --constraints_window_size 8 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc8_soft_2 --alpha 0.99

# 16x
python main.py --dataset dataset31 --loss mass_constraints --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc16_soft_0 --alpha 0.99
python main.py --dataset dataset31 --loss mass_constraints --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc16_soft_1 --alpha 0.99
python main.py --dataset dataset31 --loss mass_constraints --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints soft --batch_size 512 --model_id jmlr_wc16_soft_2 --alpha 0.99