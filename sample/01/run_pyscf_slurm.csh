#!/bin/csh -f
#
# ----------------------------------------------------------------------
#
#       Slurm job script for running python script with conda
#
#                       -ver.2.0 (2025/08/19)
#                    Mofified by Takuya Hashimoto
#
# ----------------------------------------------------------------------
#
#  <<< If you need to set the queuing system enviroinment, set here. >>>
#
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
# ----------------------------------------------------------------------
#
# ----------------------------------------------------------------------
#                            ! ATTENTION !
#       If you use this script, the following two requirements
#       must be satisfied:
#       1. Miniconda (or Anaconda) must be installed.
#       2. A virtual environment must have already been created.
#       ex) py310_pyscf_2
# ----------------------------------------------------------------------
#
#  <<< ENTER the information of your job. >>>
set DATA_DIR = `pwd`
set NAME      = sample01
set OPT_NAME  = optimize
set OUTEXT    = out
set conda_env = py310_pyscf2110
setenv OMP_NUM_THREADS 4

# show the information of your job
setenv LANG C
echo "----- Python execution script -----"
echo "Host: `hostname`"
echo "OS: `uname` at `date`"
echo "Directory: $DATA_DIR"
echo "Python script: ${NAME}.py"
echo "Conda environment: $conda_env"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Partition: $SLURM_JOB_PARTITION"
#
# check your python script
if (! -e $DATA_DIR/${NAME}.py) then
  echo "Error: Python script ${NAME}.py was not found in $DATA_DIR"
  exit 1
endif

# create the work directory & copy the script
set WORK_DIR = /work/$USER/python_job.$SLURM_JOB_ID
mkdir -pv $WORK_DIR
cp -r $DATA_DIR/* $WORK_DIR
cd $WORK_DIR

# execute python script
echo "Running python script ${NAME}.py"
python ${NAME}.py >& ${NAME}.$OUTEXT
echo "Python script was finished. Output was saved in ${NAME}.$OUTEXT"

# copy section
cp ${NAME}.$OUTEXT $DATA_DIR

# delete the work directory
cd $HOME
rm -rf $WORK_DIR
echo "Job was completed successfully at `date`"
exit

