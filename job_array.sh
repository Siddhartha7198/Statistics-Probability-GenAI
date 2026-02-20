#!/bin/bash

################
#
# Setting slurm options
#
################

# lines starting with "#SBATCH" define your jobs parameters

#SBATCH --partition graphic

##SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:tesla_v100:1
##SBATCH --gres=gpu:tesla_v100_16gb:1
#SBATCH --gres=gpu:h200:1

# telling slurm how many instances of this job to spawn (typically 1)
#SBATCH --ntasks 1

# setting number of CPUs per task (1 for serial jobs)
#SBATCH --cpus-per-task 1

# setting memory requirements
#SBATCH --mem-per-cpu 5G

# propagating max time for job to run - choose one of the formats below
#SBATCH --time 4-00:00:00

#SBATCH --array=10,50,100

# Setting the name for the job
#SBATCH --job-name nobkg

# setting notifications for job
# accepted values are ALL, BEGIN, END, FAIL, REQUEUE
#SBATCH --mail-type FAIL

# telling slurm where to write output and error
# this will create the output files in the current directory should you wish
#for them to be put elsewhere use absolute paths e.g. /home/<user>/queue/output
#SBATCH --output=/data/finite/poddar/gan/diffgan/%A_%a.out 
#SBATCH --error=/data/finite/poddar/gan/diffgan/%A_%a.out 

################
#
# copying your data to /scratch
#
################

# create local folder on ComputeNode
# ALWAYS copy any relevant data for your job to local disk to speed up your job
# and decrease load on the fileserver
scratch=/scratch/$USER/$SLURM_JOB_ID
mkdir -p $scratch
cp /data/finite/poddar/gan/diffgan/3d_den_ribosome_new.npy $scratch/3d_den.npy
cp /data/finite/poddar/gan/diffgan/gan_diff_nth.py $scratch
cp /data/finite/poddar/gan/diffgan/default.pltstyle $scratch
cd $scratch

# dont access /home after this line

# if needed load modules here
module load python/3.8.3
module load cuda/12.1

# if needed add export variables here
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


################
#
# run the program
#
################
srun python3 gan_diff_nth.py 0 0 $SLURM_ARRAY_TASK_ID


# copy results to data
mkdir -p /data/finite/poddar/gan/diffgan/ribosome/$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID
cp -ar Results/* /data/finite/poddar/gan/diffgan/ribosome/$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID


# leaving scratch
cd

# clean up scratch
rm -rf $scratch
unset scratch

exit 0

