#!/bin/bash

# Set base directories
DATA=DATA
OUTPUT=OUTPUT
TRAINER=ZeroshotCLIP

# Config for the trainer
CFG=$1       # trainer config name (e.g., vit_b16_ep50)
TEXTS=$2     # prompt strategy (e.g., cupl_full)
DATASET=$3   # dataset name (e.g., cifar10) (optional)
SEED=1

CSV_FILE="${OUTPUT}/${TRAINER}_${CFG}_${TEXTS}.csv"
echo "config,texts,dataset,shots,accuracy,accuracy_std,train_time" > "$CSV_FILE"  # initialize header (optional)

# Exclude these datasets
EXCLUDE_DATASETS=("imagenet_a" "imagenet_r" "imagenet_sketch" "imagenetv2")

# Check if DATASET is provided
if [ -z "$DATASET" ]; then
    echo "No dataset specified, running all datasets."
    DATASET_FILES="configs/datasets/*.yaml"
else
    echo "Running only the specified dataset: $DATASET"
    DATASET_FILES="configs/datasets/${DATASET}.yaml"
fi

# Filter out excluded datasets
for EXCLUDE in "${EXCLUDE_DATASETS[@]}"; do
    DATASET_FILES=$(echo "$DATASET_FILES" | grep -v "$EXCLUDE")
done

# Loop through dataset config files
for DATASET_FILE in $DATASET_FILES; do
    # Skip excluded datasets explicitly
    for EXCLUDE in "${EXCLUDE_DATASETS[@]}"; do
        if [[ "$DATASET_FILE" == *"$EXCLUDE"* ]]; then
            echo "Skipping excluded dataset: $DATASET_FILE"
            continue 2
        fi
    done

    DATASET=$(basename "$DATASET_FILE" .yaml)
    echo "Running dataset: $DATASET"

    DIR=${OUTPUT}/${DATASET}/${TRAINER}/${CFG}_csb${CSB}/${SHOTS}shots/texts_${TEXTS}/seed${SEED}

    # Remove existing output directory to force fresh run
    if [ -d "$DIR" ]; then
        echo "Removing existing directory: $DIR"
        rm -rf "$DIR"
    fi

    # Run train.py and capture output
    TRAIN_OUTPUT=$(python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file ${DATASET_FILE} \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --eval-only \
        TRAINER.SEMOBRIDGE.TEXTS ${TEXTS} \
        DATASET.NUM_SHOTS 0)

    # Extract accuracy line (first match)
    SUMMARY_LINE=$(echo "$TRAIN_OUTPUT" | grep -m1 -oP 'accuracy: \d+\.\d+%')
    ACCURACY_VAL=$(echo "$SUMMARY_LINE" | grep -oP '\d+\.\d+')

    STD_VAL="0.00"
    TIME_VAL="0:00:00"

    TEXTS_REPL="${TEXTS//,/&}"
    CSV_KEY="${CFG},${TEXTS_REPL},${DATASET},0"

    # Append to CSV if not already present
    if grep -q "^${CSV_KEY}," "$CSV_FILE"; then
        echo "Skipping CSV write: entry already exists for ${CSV_KEY}"
    else
        echo "${CSV_KEY},${ACCURACY_VAL},${STD_VAL},${TIME_VAL}" >> "$CSV_FILE"
    fi
done
