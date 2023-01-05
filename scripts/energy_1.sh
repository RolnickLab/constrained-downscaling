#!/bin/sh



# mult

# predict rho
#python main.py --dataset dataset42 --scale none --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints mult --batch_size 64 --model_id jmlr_rho4_mult_0
# predict S
python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints mult --batch_size 64 --model_id jmlr_s4_mult_1
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints mult --batch_size 64 --model_id jmlr_qv4_mult_1

python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints mult --batch_size 64 --model_id jmlr_s4_mult_2
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints mult --batch_size 64 --model_id jmlr_qv4_mult_2

#predict and evaluate Te
#python scoring.py --dataset dataset45 --test_val_train test --model_id mult_0 --model temp --factor 4
#predict and evaluate QL
#python scoring.py --dataset dataset44 --test_val_train test --model_id mult_0 --model ql --factor 4

# softmax

# predict rho
#python main.py --dataset dataset42 --scale none --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 64 --model_id jmlr_rho4_softmax_0
# predict S
python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 64 --model_id jmlr_s4_softmax_1
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 64 --model_id jmlr_qv4_softmax_1

python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 64 --model_id jmlr_s4_softmax_2
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints softmax --batch_size 64 --model_id jmlr_qv4_softmax_2

#predict and evaluate Te
#python scoring.py --dataset dataset45 --test_val_train test --model_id softmax_0 --model temp --factor 4
#predict and evaluate QL
#python scoring.py --dataset dataset44 --test_val_train test --model_id softmax_0 --model ql --factor 4

##interpolations
#python scoring.py --dataset dataset41 --test_val_train test --model_id kronecker_qv --model kronecker --factor 4
#predict and evaluate QL
#python scoring.py --dataset dataset44 --test_val_train test --model_id kronecker_ql --model kronecker --factor 4

#python scoring.py --dataset dataset43 --test_val_train test --model_id kronecker_T --model kronecker --factor 4

#python scoring.py --dataset dataset41 --test_val_train test --model_id bc_qv --model bicubic --factor 4
#predict and evaluate QL
#python scoring.py --dataset dataset44 --test_val_train test --model_id bc_ql --model bicubic --factor 4

#python scoring.py --dataset dataset43 --test_val_train test --model_id bc_T --model bicubic --factor 4