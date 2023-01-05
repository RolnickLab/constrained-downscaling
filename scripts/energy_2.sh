#!/bin/sh

# none

# predict rho
#python main.py --dataset dataset42 --scale none --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_rho4_none_0
# predict S
python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_s4_none_1
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_qv4_none_1

python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_s4_none_2
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints none --batch_size 512 --model_id jmlr_qv4_none_2

#predict and evaluate Te
#python scoring.py --dataset dataset45 --test_val_train test --model_id none_0 --model temp --factor 4
#predict and evaluate QL
#python scoring.py --dataset dataset44 --test_val_train test --model_id none_0 --model ql --factor 4

# gh

# predict rho
#python main.py --dataset dataset42 --scale none --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_rho4_gh_0
# predict S
python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_s4_gh_1
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_qv4_gh_1


python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_s4_gh_2
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints enforce_op --batch_size 512 --model_id jmlr_qv4_gh_2
#predict and evaluate Te
#python scoring.py --dataset dataset45 --test_val_train test --model_id gh_0 --model temp --factor 4
#predict and evaluate QL
#python scoring.py --dataset dataset44 --test_val_train test --model_id gh_0 --model ql --factor 4

# add

# predict rho
#python main.py --dataset dataset42 --scale none --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_rho4_add_0
# predict S
python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_s4_add_1
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_qv4_add_1

python main.py --dataset dataset43 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_s4_add_2
# predict qv
python main.py --dataset dataset41 --scale minmax --epochs 200 --upsampling_factor 4 --factor 4 --constraints_window_size 4 --model resnet2 --constraints add --batch_size 512 --model_id jmlr_qv4_add_2

#predict and evaluate Te
#python scoring.py --dataset dataset45 --test_val_train test --model_id add_0 --model temp --factor 4
#predict and evaluate QL
#python scoring.py --dataset dataset44 --test_val_train test --model_id add_0 --model ql --factor 4

# mult


