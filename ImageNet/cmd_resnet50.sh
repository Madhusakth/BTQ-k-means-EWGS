#!/bin/sh
#SBATCH -J res50           # Job name
#SBATCH -o exp_resnet_50.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gtx                           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 24:00:00              # Run time (hh:mm:ss) - 3.5 hours

python3 ImageNet_train_quant_script.py --arch=resnet50_quant --data=../data/imagenet/ --weight_levels=256 --act_levels=256  --log_dir=../results/resnet50_imagenet/W8A8_cv_6_pw_2_fc_2_bits_6_no_hessian_bs_64 --btq=True --batch-size=64 --lr_m=0.008 --workers=4 --cv_block_size=6 --pw_block_size=2 --fc_block_size=2 --bits=6 --debug=False --epochs=150 --use_hessian=False --resume=../results/resnet50_imagenet/W8A8_cv_6_pw_2_fc_2_bits_6_no_hessian_bs_64/checkpoint.pth.tar > resnet50_4.log 
