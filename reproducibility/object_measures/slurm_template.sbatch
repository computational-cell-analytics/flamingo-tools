#!/bin/bash
#SBATCH -t 24:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=constantin.pape@informatik.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007

#SBATCH -c 32
#SBATCH --mem 256G

PYTHON=/scratch-grete/usr/nimcpape/software/mamba/envs/sam/bin/python
SCRIPT=/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/reproducibility/object_measures/repro_object_measures.py

$PYTHON $SCRIPT --input $1 --output $2
