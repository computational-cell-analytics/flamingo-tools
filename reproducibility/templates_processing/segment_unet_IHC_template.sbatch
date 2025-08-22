#!/bin/bash
#SBATCH --job-name=segment-unet-SGN
#SBATCH --time 10:00:00             # for gerbil up to ~30 hours

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

# name of cochlea, as it appears in MoBIE and the NHR
COCHLEA=$1
# segmentation name, as it appears in MoBIE, e.g. IHC_v4
SEG_NAME=$2

export SCRIPT_DIR=$SCRIPT_REPO/scripts
export OUTPUT_FOLDER=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/"$COCHLEA"/"$SEG_NAME"
export MIN_SIZE=1000

# v4a
# export CENTER_DISTANCE_THRESHOLD=0.4
# export BOUNDARY_DISTANCE_THRESHOLD=0.5
# export DISTANCE_SMOOTHING=0

# v4b
export CENTER_DISTANCE_THRESHOLD=0.5
export BOUNDARY_DISTANCE_THRESHOLD=0.6
export DISTANCE_SMOOTHING=0.6

echo "Output directory: ${OUTPUT_FOLDER}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import unet_prediction;'
	'unet_prediction.run_unet_segmentation_slurm(output_folder=os.environ["OUTPUT_FOLDER"],'
	'min_size=os.environ["MIN_SIZE"])')
cmd="${cmd_array[*]}"
python -c "$cmd"

