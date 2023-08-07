# CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.launch --nproc_per_node=1 main.py --dataset_name bpae --data_path data_BPAE/train/mixed --target_path data_BPAE/train/target --batch_size 8 --model UNet --version 0 --loss L1_SSIM_MSE --crop_size 256 --channels 3
# CUDA_VISIBLE_DEVICES=1,2 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --dataset_name bpae --data_path data_BPAE/train/mixed --target_path data_BPAE/train/target --batch_size 8 --crop_size 256 --model UNet --version 1 --channels 3 --loss alpha_SSIM_L1
# python3 test.py --dataset_name bpae --dataset_name bpae --data_path data_BPAE/train/mixed --target_path data_BPAE/train/target --batch_size 4 --crop_size 256 --model UNet --version 1 --channels 3 > result/bpae_1.txt
# CUDA_VISIBLE_DEVICES=5 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --dataset_name real3 --data_path data_real3/train/mixed --target_path data_real3/train/target --batch_size 8 --crop_size 256 --model UNet --version 4 --channels 3 --loss Charbonnier --epochs 200
CUDA_VISIBLE_DEVICES=6,7 python3 -m torch.distributed.launch --nproc_per_node=2 main.py --dataset_name 3_bpae --data_path data_3_bpae/train/mixed --target_path data_3_bpae/train/target --batch_size 4 --crop_size 256 --model myNet --version 2 --channels 3 --loss all --epochs 200
# CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 main.py --dataset_name 2_bpae --data_path data_2_bpae/train/mixed --target_path data_2_bpae/train/target --batch_size 4 --crop_size 256 --model UNet --version 1 --channels 2 --loss MSE --epochs 200


# test real4
# python3 test.py --dataset_name real4 --data_path data_real4/train/mixed --target_path data_real4/train/target --crop_size 256 --model UNet --version 13 --channels 4 --loss All --augmentation
# python3 test.py --dataset_name real4 --data_path data_real4/train/lu/mixed --target_path data_real4/train/lu/target --batch_size 1 --model UNet --version 13 --channels 4 --loss All --augmentation --test_epoch 200

# test real3
# python3 test.py --dataset_name real3 --data_path data_real3/train/mixed --target_path data_real3/train/target --crop_size 256 --model UNet --version 4 --channels 3 --loss All --augmentation
# python3 test.py --dataset_name real3 --data_path data_real3/train/lu-1/mixed --target_path data_real3/train/lu-1/target --crop_size 256 --model UNet --version 4 --channels 3 --loss All --augmentation


# python3 test.py --dataset_name 3_bpae --data_path data_3_bpae/train/mixed --target_path data_3_bpae/train/target --crop_size 256 --model UwUNet --version 1 --channels 3 --loss Charbonnier --test_epoch 300 --augmentation
# python3 test.py --dataset_name 3_bpae --data_path data_3_bpae/train/mixed --target_path data_3_bpae/train/target --crop_size 256 --model UNet --version 2 --channels 3 --loss mse --test_epoch 200 --augmentation
# python3 test.py --dataset_name real3 --data_path data_3_bpae/train/lu/mixed --target_path data_3_bpae/train/lu/target --crop_size 256 --model UNet --version 2 --channels 3 --loss mse --test_epoch 300 --augmentation
# python3 test.py --dataset_name 3_bpae --data_path data_3_bpae/train/real_1126/mixed --target_path data_3_bpae/train/real_1126/target --crop_size 256 --model myNet --version 21 --channels 3 --loss all --test_epoch 200 --augmentation

# python3 main.py --dataset_name 3_bpae --data_path data_3_bpae/train/mixed --target_path data_3_bpae/train/target --batch_size 16  --crop_size 256 --model myNet --version 20 --channels 3 --loss all --epochs 200
# python3 test.py --dataset_name 3_bpae --data_path data_3_bpae/train/mixed --target_path data_3_bpae/train/target --batch_size 4  --crop_size 256 --model myNet --version 2 --channels 3 --loss all --epochs 200
# python3 test.py --dataset_name 3_bpae --data_path data_3_bpae/train/real_1126/mixed --target_path data_3_bpae/train/real_1126/target --batch_size 4 --crop_size 256 --model myNet --version 20 --channels 3 --loss all --epochs 200
python3 test.py --dataset_name 3_bpae --data_path data_simu3_group2-new/train/mixed --target_path data_simu3_group2-new/train/target --batch_size 4  --crop_size 256 --model myNet --version 21 --channels 3 --loss all --epochs 200


# CUDA_VISIBLE_DEVICES=5 python3 main.py --dataset_name real3 --data_path data_real3/train/mixed --target_path data_real3/train/target --batch_size 16  --crop_size 256 --model myNet --version 20 --channels 3 --loss all --epochs 200 --no_amp
# python3 test.py --dataset_name real3 --data_path data_real3/train/mixed --target_path data_real3/train/target --batch_size 4  --crop_size 256 --model myNet --version 20 --channels 3 --loss all --epochs 200

# CUDA_VISIBLE_DEVICES=5 python3 main.py --dataset_name real4 --data_path data_real4/train/mixed --target_path data_real4/train/target --batch_size 16  --crop_size 256 --model myNet --version 22 --channels 4 --loss all --epochs 200 --no_amp
# python3 test.py --dataset_name real4 --data_path data_real4/train/mixed --target_path data_real4/train/target --batch_size 4  --crop_size 256 --model myNet --version 20 --channels 4 --loss all --epochs 200
# python3 test.py --dataset_name real4 --data_path data_real4/train/lu/mixed --target_path data_real4/train/lu/target --batch_size 4  --crop_size 256 --model myNet --version 20 --channels 4 --loss all --test_epoch 200
python3 test.py --dataset_name real4 --data_path data_simu4_group2/train/mixed --target_path data_simu4_group2/train/target --batch_size 4  --crop_size 256 --model myNet --version 20 --channels 4 --loss all --test_epoch 200

# python3 fine_tune.py --dataset_name real4 --data_path data_real4/train/lu/mixed --target_path data_real4/train/lu/target --batch_size 4  --crop_size 256 --model myNet --version 20 --channels 4 --loss all --epochs 20
python3 fine_tune.py --dataset_name real4 --data_path data_simu4_group2/train/mixed --target_path data_simu4_group2/train/target --batch_size 1  --crop_size 256 --model myNet --version 20 --channels 4 --loss all --epochs 20 --num 35


# python3 test.py --dataset_name real3 --data_path data_real3/train/lu-1/mixed --target_path data_real3/train/lu-1/target --batch_size 4 --crop_size 256 --model myNet --version 20 --channels 3 --loss all --test_epoch 200
# python3 test.py --dataset_name real3 --data_path data_real3/train/lu/mixed --target_path data_real3/train/lu/target --batch_size 4 --crop_size 256 --model myNet --version 20 --channels 3 --loss all --test_epoch 200

# python3 test.py --dataset_name 3_bpae --data_path data_real3/train/lu/mixed --target_path data_real3/train/lu/target --batch_size 1 --crop_size 512 --model myNet --version 21 --channels 3 --loss all --test_epoch 200
python3 test.py --dataset_name 3_bpae --data_path data_real3/train/mixed --target_path data_real3/train/target --batch_size 1 --crop_size 512 --model myNet --version 21 --channels 3 --loss all --test_epoch 200
python3 test.py --dataset_name 3_bpae --data_path data_simu4_group3/train/mixed --target_path data_simu4_group3/train/target --batch_size 4 --crop_size 256 --model myNet --version 21 --channels 3 --loss all --test_epoch 200

# python3 fine_tune.py --dataset_name 3_bpae --data_path data_real3/train/lu-1/mixed --target_path data_real3/train/lu-1/target --batch_size 4 --crop_size 256 --model myNet --version 21 --channels 3 --loss all --epochs 20