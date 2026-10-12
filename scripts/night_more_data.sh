#!/usr/bin/env bash
# Suite automatique de l'expérience « plus de données » (nuit du 7 au 8 octobre).
# Attend la fin de la génération du lot 2, puis : reprise des échecs, listes, cache, entraînement, évaluation.
set -u
cd "$(dirname "$0")/.."
for f in data/smapshot/label2_gpu*.pid; do
    while kill -0 "$(cat "$f")" 2>/dev/null; do sleep 30; done
done
echo "lot 2 terminé : $(date +%T), étiquettes : $(ls data/smapshot/out/labels/*.npz | wc -l)"

# reprise des échecs (connexions coupées), un processus par GPU
for g in 0 1 2; do
    cat data/smapshot/batch2_gpu${g}a.txt data/smapshot/batch2_gpu${g}b.txt > data/smapshot/retry2_gpu$g.txt
    CUDA_VISIBLE_DEVICES=$g uv run python scripts/smapshot_labels.py data/smapshot/retry2_gpu$g.txt data/smapshot/out 1024 \
        > data/smapshot/retry2_gpu$g.log 2>&1 &
done
wait

# listes : sélection combinée, même filtre qualité ; le test d'origine reste inchangé pour comparer
uv run python -c "
import pandas as pd
pd.concat([pd.read_csv('data/smapshot/selection.csv'), pd.read_csv('data/smapshot/selection2.csv')]).to_csv('data/smapshot/selection_all.csv', index=False)"
uv run python scripts/smapshot_splits.py data/smapshot/out data/smapshot/selection_all.csv data/smapshot/lists_all
OMP_NUM_THREADS=1 uv run python scripts/build_cache.py data/smapshot/out/labels data/smapshot_cache 40

# entraînement S1 sur toutes les données (même recette que S1)
mkdir -p outputs/smap_S1_more
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc_per_node=1 --master_port=29530 scripts/train_ridges.py \
    --out outputs/smap_S1_more --cache data/smapshot_cache --train-list data/smapshot/lists_all/train.txt \
    --val-list data/smapshot/lists/val.txt --epochs 30 --compile --workers 8 > outputs/smap_S1_more/train.log 2>&1

# évaluations : même test qu'avant (comparaison directe) + CH1 + seuil des crêtes
CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_smapshot.py data/smapshot/lists/test.txt data/smapshot/out/labels \
    outputs/eval_smapshot_more outputs/smap_S1_more/best.pt outputs/smap_S1_base512/best.pt > outputs/eval_smapshot_more.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 uv run python scripts/eval_ch1.py data/ch1/extracted outputs/eval_ch1_more outputs/smap_S1_more/best.pt \
    > outputs/eval_ch1_more.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 uv run python scripts/tune_ridge_threshold.py data/smapshot/lists/val.txt data/smapshot/out/labels \
    outputs/smap_S1_more/best.pt > outputs/tune_more.log 2>&1 &
wait
echo "fini : $(date +%T)"
wc -l data/smapshot/lists_all/*.txt
grep " med " outputs/eval_smapshot_more.log outputs/eval_ch1_more.log
grep -E "meilleur|seuil 0.40" outputs/tune_more.log
