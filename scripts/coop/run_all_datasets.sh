#!/bin/bash

# Set base directories
DATA=DATA
OUTPUT=OUTPUT
TRAINER=CoOp

# Config for the trainer
CFG=$1       # trainer config name (e.g., vit_b16_ep50)
CTP=$2  # class token position (end or middle)
NCTX=$3  # number of context tokens
CSC=$4  # class-specific context (False or True)
DATASET=$5   # dataset name (e.g., cifar10) (optional)
SHOTS=$6      # number of shots (e.g., 1, 2, 4, 8, 16) (optional)

CSV_FILE="${OUTPUT}/CoOp/${CFG}_${CTP}_${NCTX}_${CSC}.csv"
echo "config,ctp,nctx,csc,dataset,shots,accuracy,accuracy_std,train_time" > "$CSV_FILE"  # initialize header (optional)

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

EXCLUDE_DATASETS=("imagenet_a" "imagenet_r" "imagenet_sketch" "imagenetv2")
# Loop through all .yaml files in dataset config folder
for DATASET_FILE in $DATASET_FILES; do
    # Check if dataset is excluded
    for EXCLUDE in "${EXCLUDE_DATASETS[@]}"; do
        if [[ "$DATASET_FILE" == *"$EXCLUDE"* ]]; then
            echo "Skipping excluded dataset: $DATASET_FILE"
            continue 2  # Skip to the next dataset file
        fi
    done

    DATASET=$(basename "$DATASET_FILE" .yaml)  # strip folder and .yaml
    echo "Running dataset: $DATASET"

    for SHOTS in $SHOTS_LIST; do
        ALL_DONE=true

        for SEED in 1 2 3; do
            DIR=${OUTPUT}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

            if [ -d "$DIR" ]; then
                echo "Skipping existing: $DIR"
            else
                ALL_DONE=false
                echo "Running: $DATASET | shots=${SHOTS} | seed=${SEED}"

                python train.py \
                    --root ${DATA} \
                    --seed ${SEED} \
                    --trainer ${TRAINER} \
                    --dataset-config-file configs/datasets/${DATASET}.yaml \
                    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
                    --output-dir ${DIR} \
                    TRAINER.COOP.N_CTX ${NCTX} \
                    TRAINER.COOP.CSC ${CSC} \
                    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                    DATASET.NUM_SHOTS ${SHOTS}
            fi
        done

        # After all 3 seeds are processed, run parse script
        if [ "$ALL_DONE" = false ]; then
            echo "Parsing test results for ${DATASET}, ${SHOTS} shots..."
        fi

        # Run and capture parse output
        PARSE_OUTPUT=$(python parse_test_res.py ${OUTPUT}/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/)

        # Extract the summary line only
        SUMMARY_LINE=$(echo "$PARSE_OUTPUT" | grep "accuracy:" | tail -n 1 | tr -d '\r\n')

        # Parse values using clean extraction
        ACCURACY_VAL=$(echo "$SUMMARY_LINE" | grep -oP '\d+\.\d+(?=%)' | sed -n '1p')
        STD_VAL=$(echo "$SUMMARY_LINE" | grep -oP '\+-\s*\K\d+\.\d+' | sed -n '1p')

        TIME_LINE=$(echo "$PARSE_OUTPUT" | grep "training_time:" | tail -n 1 | tr -d '\r\n')
        TIME_VAL=$(echo "$TIME_LINE" | awk '{print $3}')

        # Construct the unique key for this entry
        CSV_KEY="${CFG},${CTP},${NCTX},${CSC},${DATASET},${SHOTS}"

        # Check if this line already exists (match first 4 columns)
        if grep -q "^${CSV_KEY}," "$CSV_FILE"; then
            echo "Skipping CSV write: entry already exists for ${CSV_KEY}"
        else
            echo "${CSV_KEY},${ACCURACY_VAL},${STD_VAL},${TIME_VAL}" >> "$CSV_FILE"
        fi
    done
done