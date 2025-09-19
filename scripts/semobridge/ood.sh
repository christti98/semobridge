#!/bin/bash

# Set base directories
DATA=/home/damilab/semobridge/DATA
OUTPUT=OUTPUT
TRAINER=SeMoBridge

# Config for the trainer
CFG=$1       # trainer config name (e.g., vit_b16, rn50)
TEXTS=$2     # prompt strategy (e.g. aphotofa clip clip_ensemble cupl_full OR combinations like clip_ensemble,cupl_full)
CSB=$3       # Class-specific bias (True or False)
SHOTS=$4     # number of shots (e.g., 1, 2, 4, 8, 16) (optional)

CSV_FILE="${OUTPUT}/OOD/${TRAINER}_${CFG}_${TEXTS}_csb${CSB}.csv"
echo "config,texts,dataset,shots,accuracy,accuracy_std,train_time" > "$CSV_FILE"  # initialize header (optional)

# Check if DATASET is provided
if [ -z "$DATASET" ]; then
    echo "No dataset specified, running all datasets."
    DATASET_FILES="configs/datasets/*.yaml"
else
    echo "Running only the specified dataset: $DATASET"
    DATASET_FILES="configs/datasets/${DATASET}.yaml"
fi

for EXCLUDE in "${EXCLUDE_DATASETS[@]}"; do
    DATASET_FILES=$(echo "$DATASET_FILES" | grep -v "$EXCLUDE")
done

# Check if SHOTS is provided
if [ -z "$SHOTS" ]; then
    echo "No shots specified, running all shots."
    SHOTS_LIST="1 2 4 8 16"
else
    echo "Running only the specified number of shots: $SHOTS"
    SHOTS_LIST=$SHOTS
fi

#OOD_DATASETS=("imagenet_a" "imagenet_r" "imagenet_sketch" "imagenetv2")
OOD_DATASETS=("imagenet_sketch" "imagenetv2")
# Loop through all .yaml files in dataset config folder
for DATASET_FILE in $DATASET_FILES; do
    # Skip if dataset is not in OOD_DATASETS
    if [[ ! " ${OOD_DATASETS[@]} " =~ " $(basename "$DATASET_FILE" .yaml) " ]]; then
        continue
    fi

    DATASET=$(basename "$DATASET_FILE" .yaml)  # strip folder and .yaml
    echo "Running dataset: $DATASET"

    for SHOTS in $SHOTS_LIST; do
        ALL_DONE=true

        for SEED in 1 2 3; do
        #for SEED in 1; do
            MODEL_DIR=${OUTPUT}/imagenet/${TRAINER}/${CFG}_csb${CSB}/${SHOTS}shots/texts_${TEXTS}/seed${SEED}
            DIR=${OUTPUT}/${DATASET}/${TRAINER}/${CFG}_csb${CSB}/${SHOTS}shots/texts_${TEXTS}/seed${SEED}

            if [ -d "$DIR" ]; then
                echo "Skipping existing: $DIR"
                continue
            fi
            if [ ! -d "$MODEL_DIR" ]; then
                echo "Model directory does not exist: $MODEL_DIR, skipping."
                continue
            fi

            ALL_DONE=false
            echo "Running: $DATASET | shots=${SHOTS} | seed=${SEED}"

            python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file ${DATASET_FILE} \
                --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                --output-dir ${DIR} \
                --model-dir ${MODEL_DIR} \
                --eval-only \
                --load-epoch 5000 \
                TRAINER.SEMOBRIDGE.TEXTS ${TEXTS} \
                TRAINER.SEMOBRIDGE.CSB ${CSB} \
                DATASET.NUM_SHOTS ${SHOTS}
        done

        # After all 3 seeds are processed, run parse script
        if [ "$ALL_DONE" = false ]; then
            echo "Parsing test results for ${DATASET}, ${SHOTS} shots..."
        fi

        # Run and capture parse output
        PARSE_OUTPUT=$(python parse_test_res.py --test-log ${OUTPUT}/${DATASET}/${TRAINER}/${CFG}_csb${CSB}/${SHOTS}shots/texts_${TEXTS}/)

        # Extract the summary line only
        SUMMARY_LINE=$(echo "$PARSE_OUTPUT" | grep "accuracy:" | tail -n 1 | tr -d '\r\n')

        # Parse values using clean extraction
        ACCURACY_VAL=$(echo "$SUMMARY_LINE" | grep -oP '\d+\.\d+(?=%)' | sed -n '1p')
        STD_VAL=$(echo "$SUMMARY_LINE" | grep -oP '\+-\s*\K\d+\.\d+' | sed -n '1p')

        # Construct the unique key for this entry
        TEXTS_REPL="${TEXTS//,/&}"
        CSV_KEY="${CFG},${TEXTS_REPL},${DATASET},${SHOTS}"

        # Check if this line already exists (match first 4 columns)
        if grep -q "^${CSV_KEY}," "$CSV_FILE"; then
            echo "Skipping CSV write: entry already exists for ${CSV_KEY}"
        else
            echo "${CSV_KEY},${ACCURACY_VAL},${STD_VAL},0:00:00" >> "$CSV_FILE"
        fi
    done
done