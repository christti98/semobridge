#!/bin/bash

INPUT_CSV=$1
OUTPUT_CSV=$2

datasets=(
  "oxford_pets"
  "oxford_flowers"
  "fgvc_aircraft"
  "dtd"
  "eurosat"
  "stanford_cars"
  "food101"
  "sun397"
  "caltech101"
  "ucf101"
  "imagenet"
)

shots_list=(1 2 4 8 16)

# Write header if output file doesn't exist
if [[ ! -f "$OUTPUT_CSV" ]]; then
  echo "Model,Method,Shots,OxfordPets,Flowers102,FGVCAircraft,DTD,EuroSAT,StanfordCars,Food101,SUN397,Caltech101,UCF101,ImageNet,AVERAGE ACCURACY,AVERAGE TRAINING TIME" > "$OUTPUT_CSV"
fi

for shot in "${shots_list[@]}"; do
  # Extract lines with current shot
  lines=$(awk -F',' -v s="$shot" 'NR > 1 && $4 == s' "$INPUT_CSV")

  if [[ -z "$lines" ]]; then
    echo "⚠️  No data found for shot=$shot — skipping row."
    continue
  fi

  declare -A accs
  declare -A stds
  declare -A times
  config_texts=""

  while IFS=',' read -r cfg texts dataset shot_val acc std time; do
    key=$(echo "$dataset" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr ' ' '_')
    accs["$key"]=$acc
    stds["$key"]=$std

    IFS=':' read -r h m s <<< "$time"
    times["$key"]=$(printf "%01d:%02d:%02d" "$h" "$m" "$s")

    config_texts="${cfg}:${texts}"
  done <<< "$lines"

  row=",${config_texts},$shot"

  sum_acc=0
  count=0
  total_time_seconds=0

  for ds in "${datasets[@]}"; do
    if [[ -n "${accs[$ds]}" ]]; then
      acc="${accs[$ds]}"
      std="${stds[$ds]}"
      time="${times[$ds]}"

      row="$row,${acc}±${std} ${time}"

      sum_acc=$(echo "$sum_acc + $acc" | bc)
      count=$((count + 1))

      IFS=':' read -r h m s <<< "$time"
      total_time_seconds=$((total_time_seconds + 3600*h + 60*m + s))
    else
      row="$row,"
    fi
  done

  if [[ $count -eq 0 ]]; then
    echo "⚠️  No usable data for shot=$shot — skipping."
    continue
  fi

  # avg_acc=$(echo "scale=2; $sum_acc / $count" | bc)
  # avg_time_sec=$((total_time_seconds / count))
  # avg_time_fmt=$(printf "%02d:%02d:%02d" $((avg_time_sec / 3600)) $(((avg_time_sec % 3600) / 60)) $((avg_time_sec % 60)))

  # row="$row,$avg_acc,$avg_time_fmt"

  echo "$row" >> "$OUTPUT_CSV"
  echo "✅ Appended row for shot=$shot"
done
