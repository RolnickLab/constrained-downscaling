#!/bin/sh



# mult

# predict rho

python main.py --dataset dataset44 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_ql4_softmax_0


python main.py --dataset dataset44 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_ql4_softmax_1


python main.py --dataset dataset44 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 512 --model_id jmlr_ql4_softmax_2