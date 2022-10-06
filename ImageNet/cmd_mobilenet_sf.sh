#!/bin/sh
#SBATCH -J mobnethess           # Job name
#SBATCH -o exp_hess.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gtx                           # Queue name
#SBATCH -N 2                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 24:00:00              # Run time (hh:mm:ss) - 3.5 hours

python3 ImageNet_train_quant.py --arch=mobilenetv2_quant --use_hessian=False --data=../data/imagenet/  --weight_levels=256 --act_levels=16 --log_dir=../results/mobilenetv2/W8A4_weighted_1000iter_008lr_cv_3_pw_2_bits_8/ --btq=True --batch-size=128 --epochs=150 --lr_m=0.008 --workers=4 --resume=../results/mobilenetv2/W8A4_weighted_1000iter_008lr_cv_3_pw_2_bits_8/checkpoint.pth.tar > mobilenet_v211.log 
#python3 ImageNet_train_quant.py --arch=mobilenetv2_quant --use_hessian=False --data=../data/imagenet/  --weight_levels=256 --act_levels=16 --btq=True --batch-size=128 --epochs=150 --lr_m=0.008 --workers=4 > demo.log 

#python3 ImageNet_train_quant_script_1.py --arch=mobilenetv2_quant --use_hessian=True --data=../data/imagenet/  --weight_levels=256 --act_levels=256 --log_dir=../results/mobilenetv2/W8A8_weighted_005lr_cv_3_pw_2_fc_4_bits_8_bs_100_hess/ --btq=True --batch-size=25 --epochs=150 --lr_m=0.005 --workers=4 --bits=8 --cv_block_size=3 --pw_block_size=2 --fc_block_size=4 --debug=True --update_scales_every=1 --visible_gpus='0,1,2,3' > mobilenet_v2_hess.log 
