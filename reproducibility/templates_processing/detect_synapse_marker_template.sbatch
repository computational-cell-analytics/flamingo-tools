#!/bin/bash
#SBATCH --job-name=synapse-marker
#SBATCH -t 08:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -A nim00007
#SBATCH -c 8
#SBATCH --mem 128G

source ~/.bashrc
# micromamba activate micro-sam_gpu
micromamba activate sam

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

export SCRIPT_DIR=$SCRIPT_REPO/scripts

# name of cochlea, as it appears in MoBIE and the NHR
COCHLEA=$1
# image channel, e.g. CTBP2 or RibA
IMAGE_CHANNEL=$2
# segmentation name, as it appears in MoBIE, e.g. synapses_v3
IHC_SEG=$3

export INPUT_PATH="$COCHLEA"/images/ome-zarr/"$IMAGE_CHANNEL".ome.zarr
export MASK_PATH="$COCHLEA"/images/ome-zarr/"$IHC_SEG".ome.zarr

export OUTPUT_FOLDER=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/"$COCHLEA"/synapses_v3_"$IHC_SEG"

export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/Synapses/synapse_detection_model_v3.pt
export INPUT_KEY="s0"
export MAX_DISTANCE=8

echo "OUTPUT_FOLDER $OUTPUT_FOLDER"
echo "MODEL $MODEL"

SCRIPT="$SCRIPT_DIR"/synapse_marker_detection/marker_detection.py 
python $SCRIPT \
	--input "$INPUT_PATH" \
	--input_key $INPUT_KEY \
	--output_folder "$OUTPUT_FOLDER" \
	--mask "$MASK_PATH" \
	--model $MODEL \
	--max_distance $MAX_DISTANCE \
	--s3
