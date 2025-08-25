#!/bin/bash
#SBATCH --job-name=apply-unet-SGN
#SBATCH -t 02:00:00             # for gerbil up to ~3 hours

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -a 0-9

source ~/.bashrc
# micromamba activate micro-sam_gpu
micromamba activate sam

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Run the script

# SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

export SCRIPT_DIR=$SCRIPT_REPO/scripts

# name of cochlea, as it appears in MoBIE and the NHR
COCHLEA=$1
# data in n5 format, e.g. GEK11L_PV_GFP_01_fused.n5
DATA=$2
# channel in n5 folder, e.g. 0, 1, or 2
STAIN_CHANNEL=$3
# segmentation name, as it appears in MoBIE, e.g. SGN_v2 or Calb1_SGN_v2
SEG_NAME=$4

export INPUT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/"$COCHLEA"/"$DATA"

export OUTPUT_FOLDER=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/"$COCHLEA"/"$SEG_NAME"

# The default v2 model
# export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/v2_cochlea_distance_unet_SGN_supervised_2025-05-27

# Domain adapted model for MLR99L
export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/v2_domain_adaptation_mlr99l/best.pt

export PREDICTION_INSTANCES=10

export INPUT_KEY="setup$STAIN_CHANNEL/timepoint0/s0"
# export INPUT_KEY="s0"

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

