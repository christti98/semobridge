#!/bin/bash

# custom config
DATA=/home/damilab/semobridge/DATA
OUTPUT=OUTPUT
TRAINER=SeMoBridge

# Config for the trainer
CFG=$1       # trainer config name (e.g., vit_b16, rn50)
TEXTS=$2     # prompt strategy (e.g. aphotofa clip clip_ensemble cupl_full OR combinations like clip_ensemble,cupl_full)
CSB=$3       # Class-specific bias (True or False)
DATASET=$4   # dataset name (e.g., oxford_pets) (optional)
SHOTS=$5     # number of shots (e.g., 1, 2, 4, 8, 16) (optional)

#for SHOTS in 1 2 4 8 16
#do
#for SEED in 1 2 3
SEED=1

DIR=${OUTPUT}/${DATASET}/${TRAINER}/${CFG}_csb${CSB}/${SHOTS}shots/texts_${TEXTS}/seed_${SEED}
# REMOVE DIR IF EXISTS
if [ -d ${DIR} ]; then
    rm -Rf ${DIR}
fi

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --model-dir ${DIR} \
    --output-dir ${DIR} \
    --hp-search \
    TRAINER.SEMOBRIDGE.TEXTS ${TEXTS} \
    TRAINER.SEMOBRIDGE.CSB ${CSB} \
    DATASET.NUM_SHOTS ${SHOTS}