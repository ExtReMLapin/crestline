#!/usr/bin/env bash
# Toutes les évaluations en mode index, une photo sur trois par GPU (3 GPU en parallèle), puis fusion.
# Usage : scripts/eval_all_parallel.sh [suffixe]     -> outputs/eval_<jeu>_<suffixe>/report.json
set -u
cd "$(dirname "$0")/.."
TAG=${1:-opti}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True      # évite la fragmentation (tableaux de 2,4 Go)
GPUS=(0 1 2)
N=${#GPUS[@]}
for k in "${!GPUS[@]}"; do
  (
    for set in ch1 data/smapshot/lists/test.txt crosslocate; do
      name=$(basename "$set" .txt); [ "$name" = test ] && name=smapshot
      CUDA_VISIBLE_DEVICES=${GPUS[$k]} SHARD=$k/$N uv run python scripts/eval_geoloc_fast.py "$set" \
        "outputs/eval_${name}_${TAG}" 0.6:0 >> "outputs/eval_${TAG}_gpu${GPUS[$k]}.log" 2>&1
    done
  ) &
done
wait
for name in ch1 smapshot crosslocate; do
  echo "== $name"
  uv run python scripts/eval_geoloc_fast.py merge "outputs/eval_${name}_${TAG}"
done
