#!/bin/bash

MOBIE_DIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet
COCHLEA=LaVision-Mar05

export BUCKET_NAME="cochlea-lightsheet"
export SERVICE_ENDPOINT="https://s3.fs.gwdg.de"
mobie.add_remote_metadata -i $MOBIE_DIR -s $SERVICE_ENDPOINT -b $BUCKET_NAME

rclone --progress copyto "$MOBIE_DIR"/"$COCHLEA" cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA"
rclone --progress copyto "$MOBIE_DIR"/project.json cochlea-lightsheet:cochlea-lightsheet/project.json
