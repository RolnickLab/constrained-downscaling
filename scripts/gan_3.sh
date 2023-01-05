#!/bin/sh

# none



#python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints none --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_none_2

# gh

#python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints add --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_add_2

# add

#python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints enforce_op --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_gh_2

# mult

python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints mult --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_mult_2

# softmax

python main.py --model gan --noise True --ensemble True --dataset dataset28 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --constraints softmax --epochs 200 --adv_factor 0.0001 --batch_size 512 --model_id jmlr_gan_softmax_2
