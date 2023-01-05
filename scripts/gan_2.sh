#!/bin/sh

# none



# add



python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints none --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_none_1

# gh

python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints add --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_add_1

# add

python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints enforce_op --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_gh_1

# mult

python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints mult --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_mult_1

# softmax

python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints softmax --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_softmax_1

