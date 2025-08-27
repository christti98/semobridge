#!/bin/bash

# custom config
#DATA=/mnt/nas/damilab/DATA
DATA=DATA
OUTPUT=OUTPUT
TRAINER=CLIP_Adapter

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)

#for SHOTS in 1 2 4 8 16
#do
for SEED in 1 2 3
#for SEED in 1
do
    DIR=${OUTPUT}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
#done