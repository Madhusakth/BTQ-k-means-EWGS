#!/bin/sh
#SBATCH -J res34           # Job name
#SBATCH -o exp_resnet34.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gtx                           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 24:00:00              # Run time (hh:mm:ss) - 3.5 hours

python3 ImageNet_train_quant_script.py --arch=resnet34_quant --data=../data/imagenet/ --weight_levels=256 --act_levels=16  --log_dir=../results/resnet34_imagenet/W8A4_cv_2_pw_1_fc_1_bits_6_no_hessian --btq=True --batch-size=128 --lr_m=0.008 --workers=4 --cv_block_size=2 --pw_block_size=1 --fc_block_size=1 --bits=6 --debug=False --epochs=150 --use_hessian=False --resume=../results/resnet34_imagenet/W8A4_cv_2_pw_1_fc_1_bits_6_no_hessian/checkpoint.pth.tar > resnet34_5.log 
