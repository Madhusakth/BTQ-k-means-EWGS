#!/bin/sh
#SBATCH -J EWGSW4A8           # Job name
#SBATCH -o experiment_1.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gtx                           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 24:00:00              # Run time (hh:mm:ss) - 3.5 hours

#python3 ImageNet_train_quant_1.py --arch=mobilenetv2_quant --use_hessian=False --data=../data/imagenet/  --weight_levels=256 --act_levels=16 --log_dir=../results/mobilenetv2/W8A4_weighted_1000iter_008lr_cv_1_pw_2_fc_2_bits_8/ --btq=True --batch-size=128 --epochs=150 --lr_m=0.008 --workers=4 --resume=../results/mobilenetv2/W8A4_weighted_1000iter_008lr_cv_1_pw_2_fc_2_bits_8/checkpoint.pth.tar > mobilenet_v2_8.log 

python3 ImageNet_train_quant_script.py --arch=mobilenetv2_quant --use_hessian=False --data=../data/imagenet/  --weight_levels=16 --act_levels=256 --log_dir=../results/mobilenetv2/W4A8_EWGS/ --btq=False --batch-size=160 --epochs=150 --lr_m=0.008 --workers=4 --bits=8 --cv_block_size=3 --pw_block_size=2 --fc_block_size=4 --debug=False --resume=../results/mobilenetv2/W4A8_EWGS/checkpoint.pth.tar > mobilenet_v2_EWGS_6.log
