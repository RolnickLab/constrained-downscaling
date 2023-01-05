#!/bin/sh

# none

python main.py --dataset dataset33 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model time_end_to_end --constraints none --batch_size 32 --model_id jmlr_wctt_none_1 --number_channels 64 --time_steps 3

# gh
# 3am 
python main.py --dataset dataset33 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model time_end_to_end --constraints enforce_op --batch_size 32 --model_id jmlr_wctt_gh_1 --number_channels 64 --time_steps 3


# add

python main.py --dataset dataset33 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model time_end_to_end --constraints add --batch_size 32 --model_id jmlr_wctt_add_1 --number_channels 64 --time_steps 3

# mult

python main.py --dataset dataset33 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model time_end_to_end --constraints mult --batch_size 32 --model_id jmlr_wctt_mult_1 --number_channels 64 --time_steps 3

# softmax

python main.py --dataset dataset33 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model time_end_to_end --constraints softmax --batch_size 64 --model_id jmlr_wctt_softmax_1 --number_channels 64 --time_steps 3

