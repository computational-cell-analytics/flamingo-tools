#!/bin/bash
#SBATCH --job-name=mean-std-SGN
#SBATCH -t 01:00:00             # usually takes ~20 min

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007
#SBATCH -c 3
#SBATCH --mem 128G

source ~/.bashrc
micromamba activate flamingo13

# Run the script

SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
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
export SEG_CLASS="sgn"
export INPUT_KEY="setup$STAIN_CHANNEL/timepoint0/s0"

if ! [[ -f $OUTPUT_FOLDER ]] ; then
	mkdir -p "$OUTPUT_FOLDER"
fi

echo "Input directory: ${INPUT}"
echo "Output directory: ${OUTPUT_FOLDER}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import unet_prediction;'
	'unet_prediction.run_unet_prediction_preprocess_slurm(input_path=os.environ["INPUT"],'
	'input_key=os.environ["INPUT_KEY"],'
	'output_folder=os.environ["OUTPUT_FOLDER"],seg_class=os.environ["SEG_CLASS"])')
cmd="${cmd_array[*]}"
python -c "$cmd"

