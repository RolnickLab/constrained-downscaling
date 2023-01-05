#!/bin/sh

# none

#python main.py --dataset dataset33 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model time_end_to_end --constraints none --batch_size 128 --model_id jmlr_wctt_none_0 --number_channels 64

# gh

python main.py --dataset dataset33 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model time_end_to_end --constraints enforce_op --batch_size 128 --model_id jmlr_wctt_gh_0 --number_channels 64


# add

python main.py --dataset dataset33 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model time_end_to_end --constraints add --batch_size 128 --model_id jmlr_wctt_add_0 --number_channels 64




#kronecker bicubic

