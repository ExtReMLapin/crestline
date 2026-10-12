#!/usr/bin/env bash
# Mesure le débit d'entraînement (images/s) de plusieurs configurations.
# Usage : bash scripts/bench_train.sh > outputs/bench.log
set -u
cd "$(dirname "$0")/.."
OUT=outputs/bench_tmp
STEPS=100

run() {  # run <gpus> <nproc> <args...>
    local gpus=$1 nproc=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpus uv run torchrun --nproc_per_node=$nproc --master_port=29600 \
        scripts/train_ridges.py --out $OUT --bench $STEPS "$@" 2>&1 | grep BENCH
}

run 0 1 --batch 8
run 0 1 --batch 8 --compile
run 0 1 --batch 16 --compile
run 0,1,2 3 --batch 16 --compile
run 0,1,2 3 --batch 16 --compile --grad-bf16
rm -rf $OUT
