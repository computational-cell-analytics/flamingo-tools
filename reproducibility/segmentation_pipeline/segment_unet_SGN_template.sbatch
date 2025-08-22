#!/bin/bash
#SBATCH --job-name=segment-unet-SGN_M-LR-000214-L
#SBATCH --time 10:00:00             # estimated time, adapt to your needs

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007
#SBATCH -c 8
#SBATCH --mem 400G

source ~/.bashrc
micromamba activate micro-sam_gpu

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Run the script
#python myprogram.py $SLURM_ARRAY_TASK_ID

SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

COCHLEA=$1

export SCRIPT_DIR=$SCRIPT_REPO/scripts
export OUTPUT_FOLDER=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/"$COCHLEA"/SGN_v2
export MIN_SIZE=1000
export BOUNDARY_DISTANCE_THRESHOLD=0.5

echo "Output directory: ${OUTPUT_FOLDER}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import unet_prediction;'
	'unet_prediction.run_unet_segmentation_slurm(output_folder=os.environ["OUTPUT_FOLDER"],'
	'min_size=os.environ["MIN_SIZE"])')
cmd="${cmd_array[*]}"
python -c "$cmd"

