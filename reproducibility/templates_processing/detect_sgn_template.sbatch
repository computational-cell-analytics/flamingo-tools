#!/bin/bash
#SBATCH --job-name=synapse-detect
#SBATCH -t 03:00:00             # estimated time, adapt to your needs
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -A nim00007
#SBATCH -c 4
#SBATCH --mem 32G

source ~/.bashrc
# micromamba activate micro-sam_gpu
micromamba activate sam

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Run the script
#python myprogram.py $SLURM_ARRAY_TASK_ID

# SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

export SCRIPT_DIR=$SCRIPT_REPO/scripts

# name of cochlea, as it appears in MoBIE and the NHR
COCHLEA=$1
# model of SGN detection, e.g. v5b
MODEL_VERSION=$2

# data on NHR
MOBIE_DIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/
export INPUT_PATH="$MOBIE_DIR"/"$COCHLEA"/images/ome-zarr/PV.ome.zarr

# data on MoBIE
# export INPUT_PATH="$COCHLEA"/images/ome-zarr/PV.ome.zarr
# use --s3 flag for script

export OUTPUT_FOLDER=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/"$COCHLEA"/SGN_detect-"$MODEL_VERSION"

if ! [[ -f $OUTPUT_FOLDER ]] ; then
	mkdir -p "$OUTPUT_FOLDER"
fi

export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/sgn-detection-"$MODEL_VERSION".pt
INPUT_KEY="s0"

echo "OUTPUT_FOLDER $OUTPUT_FOLDER"
echo "MODEL $MODEL"

python ~/flamingo-tools/scripts/sgn_detection/sgn_detection.py \
	--input "$INPUT_PATH" \
	--input_key $INPUT_KEY \
	--output_folder "$OUTPUT_FOLDER" \
	--model "$MODEL"
