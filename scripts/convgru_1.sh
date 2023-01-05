#!/bin/sh

# none

python main.py --dataset dataset32 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model conv_gru_det --constraints none --batch_size 128 --model_id jmlr_wct_none_0 --number_channels 64 --time_steps 3

# gh

python main.py --dataset dataset32 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model conv_gru_det --constraints enforce_op --batch_size 128 --model_id jmlr_wct_gh_0 --number_channels 64 --time_steps 3


# add

python main.py --dataset dataset32 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model conv_gru_det --constraints add --batch_size 128 --model_id jmlr_wct_add_0 --number_channels 64 --time_steps 3


# mult

python main.py --dataset dataset32 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model conv_gru_det --constraints mult --batch_size 128 --model_id jmlr_wct_mult_0 --number_channels 64 --time_steps 3

# softmax

python main.py --dataset dataset32 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model conv_gru_det --constraints softmax --batch_size 128 --model_id jmlr_wct_softmax_0 --number_channels 64 --time_steps 3

#kronecker bicubic

