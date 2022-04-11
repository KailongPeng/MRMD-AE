#!/usr/bin/env bash
#SBATCH --output=logs/%A_%a.out
#SBATCH --job-name usefunc2stand
#SBATCH --partition=psych_day,psych_scavenge,psych_week,day,week
#SBATCH --time=2:00:00 #20:00:00
#SBATCH --mem=10GB
##SBATCH -n 5
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kp578
set -e
cd /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt

sub=$1
echo SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID
echo sub=${sub}

echo python -u /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/usefunc2stand.py $SLURM_ARRAY_TASK_ID ${sub}
python -u /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/usefunc2stand.py $SLURM_ARRAY_TASK_ID ${sub}
