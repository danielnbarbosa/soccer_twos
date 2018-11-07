#!/bin/bash

# Sync weights from AWS and run evaluation on them.
# Example: ./scripts/evaluate.sh 54.234.29.14 run01 episode.100
# Useful for training an agent on AWS and evaluating it locally

IP=$1
LOCAL_DIR=$2
FILE_PREFIX=$3

mkdir -p checkpoints/$LOCAL_DIR/
scp ubuntu@$IP:angela/checkpoints/last_run/$FILE_PREFIX.*  checkpoints/$LOCAL_DIR/
./main.py --eval --load=checkpoints/$LOCAL_DIR/$FILE_PREFIX
