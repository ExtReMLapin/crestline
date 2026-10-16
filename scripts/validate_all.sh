#!/usr/bin/env bash
# Validation complète d'une version : mode index (3 GPU en parallèle) puis mode brute (3 GPU, un processus).
# Usage : scripts/validate_all.sh [suffixe]   -> outputs/eval_*_<suffixe>, outputs/geoloc_{ch1,smap}_brute_<suffixe>
set -u
cd "$(dirname "$0")/.."
TAG=${1:-v2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DBS="data/horizon/ch125_band0 data/horizon/ch125_band1 data/horizon/ch125_band2"
CKPT=outputs/smap_S1_more/best.pt
echo "== mode index (3 GPU en parallèle) $(date +%T)"
scripts/eval_all_parallel.sh "$TAG"
echo "== CH1 mode brute $(date +%T)"
uv run python scripts/eval_geoloc_ch1.py data/ch1/extracted "outputs/geoloc_ch1_brute_$TAG" $CKPT 100 $DBS \
  > "outputs/geoloc_ch1_brute_$TAG.log" 2>&1
tail -12 "outputs/geoloc_ch1_brute_$TAG.log"
echo "== smapshot mode brute $(date +%T)"
uv run python scripts/eval_geoloc_smapshot.py data/smapshot/lists/test.txt data/smapshot/out/photos \
  "outputs/geoloc_smap_brute_$TAG" $CKPT 100 $DBS > "outputs/geoloc_smap_brute_$TAG.log" 2>&1
tail -12 "outputs/geoloc_smap_brute_$TAG.log"
echo "== fini $(date +%T)"
