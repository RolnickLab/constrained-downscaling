#!/bin/sh

# predict rho
python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_wc16_mult_0

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_wc16_mult_1

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints mult --batch_size 512 --model_id jmlr_wc16_mult_2

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_wc16_softmax_0

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_wc16_softmax_1

python main.py --dataset dataset31 --scale minmax --epochs 200 --upsampling_factor 16 --factor 16 --constraints_window_size 16 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_wc16_softmax_2
