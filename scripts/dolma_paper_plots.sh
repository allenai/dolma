#!/usr/bin/env bash

# get script directory
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  # if $SOURCE was a relative symlink, we need to resolve it
  # relative to the path where the symlink file was located
  [[ $SOURCE != /* ]] && SOURCE="$SCRIPT_DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

train_metrics=(
    'train/Perplexity'
    'train/CrossEntropyLoss'
)
TRAIN_METRICS="$(printf "%s " "${train_metrics[@]}" | sed 's/ $//')"

eval_metrics=(
    'eval/downstream/hellaswag_len_norm'
    'eval/downstream/piqa_len_norm'
    'eval/downstream/arc_easy_acc'
    'eval/downstream/sciq_acc'
    'eval/downstream/mrpc_f1'
    'eval/downstream/commitment_bank_acc'
    'eval/downstream/copa_acc'
    'eval/downstream/openbook_qa_len_norm'
    'eval/downstream/winogrande_acc'
    'eval/downstream/rte_len_norm'
)
EVAL_METRICS="$(printf "%s " "${eval_metrics[@]}" | sed 's/ $//')"

og_perplexity_suite=(
    'eval/gab-validation/Perplexity'
    'eval/ice-validation/Perplexity'
    'eval/ptb-validation/Perplexity'
    'eval/pile-validation/Perplexity'
    'eval/4chan-validation/Perplexity'
    'eval/c4_en-validation/Perplexity'
    'eval/mc4_en-validation/Perplexity'
    'eval/m2d2_wiki-validation/Perplexity'
    'eval/m2d2_s2orc-validation/Perplexity'
    'eval/manosphere-validation/Perplexity'
    'eval/twitterAEE-validation/Perplexity'
    'eval/wikitext_103-validation/Perplexity'
    'eval/c4_100_domains-validation/Perplexity'
)
OG_PERPLEXITY_SUITE="$(printf "%s " "${og_perplexity_suite[@]}" | sed 's/ $//')"

runs_up_to_150b=(
    'olmo-small-rpj-*'
    'olmo-small-pile-fixed-*'
    'olmo-small-dolma-*'
    'olmo-small-falcon-*'
)
RUNS_UP_TO_150B="$(printf "%s " "${runs_up_to_150b[@]}" | sed 's/ $//')"

SAMPLES=1000

set -ex

python ${SCRIPT_DIR}/wandb_to_plot.py \
    -t ai2-llm \
    -p olmo-small \
    -n $RUNS_UP_TO_150B \
    -y $TRAIN_METRICS \
    -s $SAMPLES \
    -d "${1}/150B_runs/train" \
    --max-x-axis '150e9' \
    --max-y-axis 100 \
    --y-log-scale \
    -v ${SCRIPT_DIR}/wandb_run_vocab.yaml


python ${SCRIPT_DIR}/wandb_to_plot.py \
    -t ai2-llm \
    -p olmo-small \
    -n $RUNS_UP_TO_150B \
    -y $EVAL_METRICS \
    -s $SAMPLES \
    -d "${1}/150B_runs/downstream" \
    --max-x-axis '150e9' \
    -v ${SCRIPT_DIR}/wandb_run_vocab.yaml


python ${SCRIPT_DIR}/wandb_to_plot.py \
    -t ai2-llm \
    -p olmo-small \
    -n $RUNS_UP_TO_150B \
    -y $OG_PERPLEXITY_SUITE \
    -s $SAMPLES \
    -d "${1}/150B_runs/ppl" \
    --max-x-axis '150e9' \
    --max-y-axis 100 \
    --y-log-scale \
    -v ${SCRIPT_DIR}/wandb_run_vocab.yaml


# python ${SCRIPT_DIR}/wandb_to_plot.py \
#     -t ai2-llm \
#     -p olmo-small \
#     -n 'olmo-small-3T-lower-lr-tie-*' \
#     -y ${METRICS} ${OG_PERPLEXITY_SUITE} \
#     -s ${SAMPLES} \
#     -d "${1}/150B_runs" \
#     --max-x-axis '150e9' \
#     -v ${SCRIPT_DIR}/wandb_run_vocab.yaml
