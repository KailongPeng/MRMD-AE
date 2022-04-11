#!/usr/bin/env bash
#SBATCH --output=logs/%A_%a.out
#SBATCH --job-name func2template
#SBATCH --partition=psych_day,psych_scavenge,psych_week,day,week
#SBATCH --time=2:00:00 #20:00:00
##SBATCH --mem=10000
##SBATCH -n 5
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kp578
set -e
cd /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE ; module load FSL ; source /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.3-centos7_64/etc/fslconf/fsl.sh ; . /gpfs/milgram/apps/hpc.rhel7/software/Python/Anaconda3/etc/profile.d/conda.sh ; conda activate /gpfs/milgram/project/turk-browne/kp578/conda_envs/rtSynth_rt
echo SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID
echo python -u /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/convert_func_to_templateSpace.py $SLURM_ARRAY_TASK_ID
python -u /gpfs/milgram/project/turk-browne/users/kp578/localize/MRMD-AE/dataPreparation/convert_func_to_templateSpace.py $SLURM_ARRAY_TASK_ID
