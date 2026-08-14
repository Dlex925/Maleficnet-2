#!/usr/bin/env bash
set -euo pipefail
GAMMA=1e-5
MODEL=vgg16; DATASET=imagenet; DIM=224; NC=1000
PAYLOADS=(payload.exe)
for p in "${PAYLOADS[@]}"; do
    post="checkpoints/${MODEL}_${DATASET}_${p%.*}_model.pt"
    if [[ ! -f "$post" ]]; then
        echo "== inject $MODEL $p =="
        python maleficnet.py -m "$MODEL" --dataset "$DATASET" --dim "$DIM" --num_classes "$NC" --payload "$p" --gamma "$GAMMA" --epochs 0 --only_pretrained
    else
        echo "== skip inject ($post există) =="
    fi
    echo "== prune-attack $MODEL $p =="
    python attack_prune.py -m "$MODEL" --dataset "$DATASET" --dim "$DIM" --num_classes "$NC" --payload "$p" --only_pretrained
done
