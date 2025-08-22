#!/bin/bash
#SBATCH --job-name=apply-unet-SGN
#SBATCH -t 01:00:00             # estimated time, adapt to your needs

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 1
#SBATCH --mem 24G
#SBATCH -a 0-9

source ~/.bashrc
micromamba activate micro-sam_gpu

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Run the script

SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

export SCRIPT_DIR=$SCRIPT_REPO/scripts

COCHLEA=$1

export INPUT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/"$COCHLEA"/MLR214L_Calb1_PV_CR_012_fused.n5

export OUTPUT_FOLDER=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/"$COCHLEA"/SGN_v2

export MODEL=/mnt/lustre-grete/usr/u15000/checkpoints/cochlea_distance_unet_SGN_supervised_2025-05-27/
export PREDICTION_INSTANCES=10
export INPUT_KEY="setup1/timepoint0/s0"

echo "Input directory: ${INPUT}"
echo "Output directory: ${OUTPUT_FOLDER}"
echo "Model: ${MODEL}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import unet_prediction;'
	'unet_prediction.run_unet_prediction_slurm(input_path=os.environ["INPUT"],'
	'output_folder=os.environ["OUTPUT_FOLDER"],model_path=os.environ["MODEL"],'
	'input_key=os.environ["INPUT_KEY"],'
	'prediction_instances=os.environ["PREDICTION_INSTANCES"])')
cmd="${cmd_array[*]}"
python -c "$cmd"

