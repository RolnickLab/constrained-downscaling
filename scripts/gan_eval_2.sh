#!/bin/sh






python evaluate.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints none --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_none_0

# gh

python evaluate.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints add --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_add_0

# add

python evaluate.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints enforce_op --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_gh_0

# mult

