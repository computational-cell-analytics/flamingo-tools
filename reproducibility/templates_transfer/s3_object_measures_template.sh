#!/bin/bash

MOBIE_DIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet
SEG_TABLE=PV-SGN-v2
SEG_S3=PV_SGN_v2

COCHLEA_TABLE=M-AMD-N62-L
COCHLEA_S3=M_AMD_N62_L
STAINS=("Calb1" "CR")

#COCHLEA_TABLE=M-AMD-Runx1-L
#COCHLEA_S3=M_AMD_Runx1_L
#STAINS=("CR" "Ntng1")

#COCHLEA_TABLE=M-LR-000214-L
#COCHLEA_S3=M_LR_000214_L
#STAINS=("CR" "Calb1")

for stain in "${STAINS[@]}" ; do
	# use --dry-run for testing
	rclone  --progress copyto "$MOBIE_DIR"/tables/measurements2/"$COCHLEA_TABLE"_"$stain"_"$SEG_TABLE"_object-measures.tsv cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA_S3"/tables/"$SEG_S3"/"$stain"_"$SEG_TABLE"_object-measures.tsv
done

