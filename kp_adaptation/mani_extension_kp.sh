#!/usr/bin/env bash
#SBATCH --output=logs/%j.out
##SBATCH --output=logs/%A_%a.out
#SBATCH --job-name training
#SBATCH --partition=psych_day,psych_scavenge,psych_week,day,week
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kp578

# cd /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/
# sbatch ./kp_adaptation/mani_extension_kp.sh
set -e
cd /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/ ; module load miniconda ; conda activate env_fmri

ROI=early_visual # 感兴趣的区域 region of interest
BS=64 # 批量大小 batch size
NEPOCHS=10 # 纪元数 number of epochs  4000
SAVEFREQ=100 # 检查点保存频率 checkpoint save frequency
HIDDIM=64 # 共同潜伏层尺寸  common latent layer dimension
ZDIM=20 # 隐蔽维度 hidden dimension
LAMMANI=100  # 流形正则化 manifold regularization
LAM=100 # 共用嵌入层正则化 commom embedding layer regularization
TRAINPCT='4run'

python3 -u ./kp_adaptation/mani_extension_kp.py \
--train_percent=$TRAINPCT \
--ROI=$ROI \
--hidden_dim=$HIDDIM \
--zdim=$ZDIM \
--batch_size=$BS \
--lam=$LAM \
--lam_mani=$LAMMANI \
--n_epochs=$NEPOCHS

#  --consecutive_time # 只有当我们希望训练-测试时间是连续的时候。 only when we want the train-test time to be consecutive
#  --volsurf=$VOLSURF
#    python3 mani_extension.py \
#    --train_percent=$TRAINPCT \  # 用多大比例的数据作为训练数据
#    --ROI=$ROI \ # 对于哪一个脑区感兴趣
#    --hidden_dim=$HIDDIM \ # 共享潜伏层的维度大小
#    --zdim=$ZDIM \ # 隐藏层的维度
#    --volsurf=$VOLSURF \ #
#    --batch_size=$BS \ 批量大小
#    --lam=$LAM \ # 共享嵌入层的正则化
#    --lam_mani=$LAMMANI \ # 流形正则化
#    --consecutive_time # 只有当我们希望训练-测试时间是连续的时候。 only when we want the train-test time to be consecutive
#    "--train_percent=50 \
#        --ROI=early_visual \
#        --hidden_dim=64 \
#        --zdim=20 \
#        --batch_size=64 \
#        --lam=0 \
#        --lam_mani=100 \
#        --consecutive_time"
# done
# done
echo done