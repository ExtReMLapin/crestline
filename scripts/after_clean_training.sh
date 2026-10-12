#!/usr/bin/env bash
# Après l'entraînement « licences propres » : évaluations, seuil des crêtes, prédictions sur les photos.
# Attente par PID uniquement (jamais par motif de texte, qui peut correspondre à ce script lui-même).
set -u
cd "$(dirname "$0")/.."
OUT=outputs/crestline_clean
while kill -0 "$(cat $OUT/pid)" 2>/dev/null; do sleep 30; done

CUDA_VISIBLE_DEVICES=1 uv run python scripts/eval_ch1.py data/ch1/extracted outputs/eval_ch1_clean_model $OUT/best.pt \
    > outputs/eval_ch1_clean_model.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 uv run python scripts/tune_ridge_threshold.py data/smapshot/lists_clean/val.txt data/smapshot/out/labels \
    $OUT/best.pt > outputs/tune_clean.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_smapshot.py data/smapshot/lists/test.txt data/smapshot/out/labels \
    outputs/eval_smapshot_clean_model $OUT/best.pt > outputs/eval_smapshot_clean_model.log 2>&1
wait $P1 $P2
CUDA_VISIBLE_DEVICES=0 uv run python scripts/predict_photos.py $OUT/best.pt photos $OUT/fiz_2048 2048 > /dev/null 2>&1

grep -E "^epoch" $OUT/train.log | tail -1
grep " med " outputs/eval_ch1_clean_model.log outputs/eval_smapshot_clean_model.log
grep -E "meilleur|seuil 0.40" outputs/tune_clean.log
ls $OUT/fiz_2048 | wc -l
