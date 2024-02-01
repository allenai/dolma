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

code_perplexity_suite=(
    'eval/openai_humaneval_test/Perplexity'
    'eval/mbpp_valid/Perplexity'
    'eval/stack_v2_held_out/Perplexity'
)
CODE_PERPLEXITY_SUITE="$(printf "%s " "${code_perplexity_suite[@]}" | sed 's/ $//')"

v1_perplexity_suite=(
    'eval/4chan-validation/Perplexity'
    'eval/c4_100_domains-validation/Perplexity'
    'eval/c4_en-validation/Perplexity'
    'eval/gab-validation/Perplexity'
    'eval/ice-validation/Perplexity'
    'eval/m2d2_s2orc-validation/Perplexity'
    'eval/m2d2_wiki-validation/Perplexity'
    'eval/manosphere-validation/Perplexity'
    'eval/mc4_en-validation/Perplexity'
    'eval/pile-validation/Perplexity'
    'eval/twitterAEE-validation/Perplexity'
)
V1_PERPLEXITY_SUITE="$(printf "%s " "${v1_perplexity_suite[@]}" | sed 's/ $//')"

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

v2_v3_perplexity_suite=(
    'eval/v3-small-ice-validation/Perplexity'
    'eval/v3-small-dolma_wiki-validation/Perplexity'
    'eval/v3-small-dolma_books-validation/Perplexity'
    'eval/v3-small-dolma_pes2o-validation/Perplexity'
    'eval/v3-small-dolma_stack-validation/Perplexity'
    'eval/v3-small-dolma_reddit-validation/Perplexity'
    'eval/v3-small-dolma_common-crawl-validation/Perplexity'
    'eval/v2-small-gab-validation/Perplexity'
    'eval/v2-small-ptb-validation/Perplexity'
    'eval/v2-small-pile-validation/Perplexity'
    'eval/v2-small-4chan-validation/Perplexity'
    'eval/v2-small-c4_en-validation/Perplexity'
    'eval/v2-small-mc4_en-validation/Perplexity'
    'eval/v2-small-m2d2_wiki-validation/Perplexity'
    'eval/v2-small-m2d2_s2orc-validation/Perplexity'
    'eval/v2-small-manosphere-validation/Perplexity'
    'eval/v2-small-twitterAEE-validation/Perplexity'
    'eval/v2-small-wikitext_103-validation/Perplexity'
    'eval/v2-small-c4_100_domains-validation/Perplexity'
)
V2_V3_PERPLEXITY_SUITE="$(printf "%s " "${v2_v3_perplexity_suite[@]}" | sed 's/ $//')"

runs_up_to_150b=(
    'olmo-small-rpj-*'
    'olmo-small-pile-fixed-*'
    'olmo-small-c4-*'
    'olmo-small-mc4-*'
    'olmo-small-falcon-*'
    'olmo-small-dolma-*'
)
RUNS_UP_TO_150B="$(printf "%s " "${runs_up_to_150b[@]}" | sed 's/ $//')"

# base directory for all plots and points to sample
SAMPLES=1000
BASE_DIR="${1}"

###############################################################################

# Figure 1: comparison of training curves between diff datasets.
FIGURE_1_DIR="${BASE_DIR}/150b_runs"

if [ ! -d "${FIGURE_1_DIR}/train" ]; then
    # only plot if the directory doesn't exist
    set -ex

    python ${SCRIPT_DIR}/wandb_to_plot.py \
        -t ai2-llm \
        -p olmo-small \
        -n $RUNS_UP_TO_150B \
        -y $TRAIN_METRICS \
        -s $SAMPLES \
        -d ${FIGURE_1_DIR}/train \
        --max-x-axis '150e9' \
        --max-y-axis 100 \
        --y-log-scale \
        -v ${SCRIPT_DIR}/wandb_run_vocab.yaml

    set +ex
fi

if [ ! -d "${FIGURE_1_DIR}/downstream" ]; then
    # only plot if the directory doesn't exist
    set -ex

    python ${SCRIPT_DIR}/wandb_to_plot.py \
        -t ai2-llm \
        -p olmo-small \
        -n $RUNS_UP_TO_150B \
        -y $EVAL_METRICS \
        -s $SAMPLES \
        -d "${FIGURE_1_DIR}/downstream" \
        --max-x-axis '150e9' \
        -v ${SCRIPT_DIR}/wandb_run_vocab.yaml

    set +ex

fi

if [ ! -d "${FIGURE_1_DIR}/ppl" ]; then
    # only plot if the directory doesn't exist
    set -ex

    python ${SCRIPT_DIR}/wandb_to_plot.py \
        -t ai2-llm \
        -p olmo-small \
        -n $RUNS_UP_TO_150B \
        -y $OG_PERPLEXITY_SUITE \
        -s $SAMPLES \
        -d "${FIGURE_1_DIR}/ppl" \
        --max-x-axis '150e9' \
        --max-y-axis 100 \
        --y-log-scale \
        -v ${SCRIPT_DIR}/wandb_run_vocab.yaml

    set +ex
fi

###############################################################################

# Figure 2: long 1b run (up to 3T tokens if possible)
FIGURE_2_DIR="${BASE_DIR}/long_1b_run"

if [ ! -d "${FIGURE_2_DIR}/train" ]; then
    # only plot if the directory doesn't exist
    set -ex

    python ${SCRIPT_DIR}/wandb_to_plot.py \
        -t ai2-llm \
        -p olmo-small \
        -n 'olmo-small-3T-lower-lr-tie_*' \
        -y $TRAIN_METRICS \
        -s $SAMPLES \
        -d ${FIGURE_2_DIR}/train \
        --max-y-axis 100 \
        --y-log-scale \
        -v ${SCRIPT_DIR}/wandb_run_vocab.yaml

    set +ex
fi

if [ ! -d "${FIGURE_2_DIR}/downstream" ]; then
    # only plot if the directory doesn't exist
    set -ex

    python ${SCRIPT_DIR}/wandb_to_plot.py \
        -t ai2-llm \
        -p olmo-small \
        -n 'olmo-small-3T-lower-lr-tie_*' \
        -y $EVAL_METRICS \
        -s $SAMPLES \
        -d "${FIGURE_2_DIR}/downstream" \
        -v ${SCRIPT_DIR}/wandb_run_vocab.yaml

    set +ex

fi

if [ ! -d "${FIGURE_2_DIR}/ppl" ]; then
    # only plot if the directory doesn't exist
    set -ex

    python ${SCRIPT_DIR}/wandb_to_plot.py \
        -t ai2-llm \
        -p olmo-small \
        -n 'olmo-small-3T-lower-lr-tie_*' \
        -y $V2_V3_PERPLEXITY_SUITE \
        -s $SAMPLES \
        -d "${FIGURE_2_DIR}/ppl" \
        --max-y-axis 100 \
        --y-log-scale \
        -v ${SCRIPT_DIR}/wandb_run_vocab.yaml

    set +ex
fi


###############################################################################




ablations_runs=(
    'v1-small-pi-less-than-5-anonymize_* v1-small-all-pi-removed_* abl-cc-v1-small-dedup_*'
    'reddit-v1-ablation-base_* reddit-v1-ablation-pii-nsfw-toxic_filtered_* reddit-v1-ablation-toxic-filtered_*'
    'olmo-mix-v1-sample_* olmo-mix-v1-sample-all-cc* olmo-mix-v1-sample-mix2_* olmo-mix-v1-gopher-like_*'
    'stack-v2* stack-v4*'
    'GPT-Neox-20B* c4-stack-15p* c4_p85-stack_v4_p15* c4_p85-starcoder_p15*'
    'v1-small-hatespeech-filtered-low* v1-small-nsfw-filtered-low* v1-small-hatespeech-filtered-high* v1-small-nsfw-filtered-high* abl-cc-v1-small-dedup_*'
    'abl-cc-v1-small-dedup_* abl-cc-v2-small-dedup*'
    'abl-cc-v1-small-dedup_* v1-small-c4-cleaned_* v1-small-c4-filtered_* v1-small-gopher-filtered_* v1-small-c4-cleaned-gopher-filtered_* v1-small-c4-cleaned-gopher-filtered-deduped_* olmo-mix-v1-sample-all-cc*'
    'abl-cc-v1-small-dedup_* v1-small-c4-cleaned_* v1-small-c4-filtered_* v1-small-gopher-filtered_* v1-small-c4-cleaned-gopher-filtered_*'
    'abl-cc-v1-small-dedup_* v1-small-c4-cleaned-gopher-filtered_* v1-small-c4-cleaned-gopher-filtered-deduped_* olmo-mix-v1-sample-all-cc*'
    'reddit-v5-ablation-filtered-gen-2_* reddit-v3-ablation-base-* reddit-v2-ablation-base-* reddit-v4-ablation-base-* reddit-v1-ablation-base_*'
)
ablations_names=(
    'cc_pii_filtering'
    'reddit_toxic_filtering'
    'dolma_mix'
    'code_stack_v2_vs_v4'
    'code_15p_stack_v2_v4_starcoder'
    'cc_toxic_filtering'
    'cc_dedupe'
    'cc_quality'
    'cc_quality_only'
    'cc_to_quality_plus_content'
    'reddit_selection'
)

limits=(
    '150e9'
    '60e9'
    '150e9'
    '50e9'
    '50e9'
    '150e9'
    '150e9'
    '150e9'
    '150e9'
    '150e9'
    '60e9'
)

# Loop through the indices of the array.
for index in "${!ablations_names[@]}"; do
    # Access the element by its index.
    ABLATION_NAME="${ablations_names[$index]}"
    ABLATION_DIR="${BASE_DIR}/ablations_${ABLATION_NAME}"
    RUNS_ABLATION="${ablations_runs[$index]}"
    X_AXIS_LIMIT="${limits[$index]}"

    if [ ! -d "${ABLATION_DIR}/train" ]; then
        # only plot if the directory doesn't exist
        set -ex

        python ${SCRIPT_DIR}/wandb_to_plot.py \
            -t ai2-llm \
            -p c4-small \
            -n $RUNS_ABLATION \
            -N $ABLATION_NAME \
            -y $TRAIN_METRICS \
            -s $SAMPLES \
            -d ${ABLATION_DIR}/train \
            --max-x-axis $X_AXIS_LIMIT \
            --max-y-axis 100 \
            --y-log-scale \
            -v ${SCRIPT_DIR}/wandb_run_vocab.yaml \
            --plotly-font-size 9 \
            --plotly-figure-width 400 \
            --plotly-figure-height 250

        set +ex
    fi

    if [ ! -d "${ABLATION_DIR}/downstream" ]; then
        # only plot if the directory doesn't exist
        set -ex

        python ${SCRIPT_DIR}/wandb_to_plot.py \
            -t ai2-llm \
            -p c4-small \
            -n $RUNS_ABLATION \
            -N $ABLATION_NAME \
            -y $EVAL_METRICS \
            -s $SAMPLES \
            -d "${ABLATION_DIR}/downstream" \
            --max-x-axis $X_AXIS_LIMIT \
            -v ${SCRIPT_DIR}/wandb_run_vocab.yaml \
            --plotly-font-size 9 \
            --plotly-figure-width 400 \
            --plotly-figure-height 250

        set +ex

    fi

    if [ ! -d "${ABLATION_DIR}/ppl" ]; then
        # only plot if the directory doesn't exist
        set -ex

        python ${SCRIPT_DIR}/wandb_to_plot.py \
            -t ai2-llm \
            -p c4-small \
            -n $RUNS_ABLATION \
            -N $ABLATION_NAME \
            -y $V1_PERPLEXITY_SUITE \
            -s $SAMPLES \
            -d "${ABLATION_DIR}/ppl" \
            --max-x-axis $X_AXIS_LIMIT \
            --max-y-axis 100 \
            --y-log-scale \
            -v ${SCRIPT_DIR}/wandb_run_vocab.yaml \
            --plotly-font-size 9 \
            --plotly-figure-width 400 \
            --plotly-figure-height 250

        set +ex
    fi

    if ([[ $ABLATION_NAME == *"code"* ]] || [[ $ABLATION_NAME == *"dolma"* ]]) && [[ ! -d "$ABLATION_DIR/code" ]]; then
        # only plot if the directory doesn't exist
        set -ex

        python ${SCRIPT_DIR}/wandb_to_plot.py \
            -t ai2-llm \
            -p c4-small \
            -n $RUNS_ABLATION \
            -N $ABLATION_NAME \
            -y $CODE_PERPLEXITY_SUITE \
            -s $SAMPLES \
            -d "${ABLATION_DIR}/code" \
            --max-x-axis $X_AXIS_LIMIT \
            --max-y-axis 100 \
            --y-log-scale \
            -v ${SCRIPT_DIR}/wandb_run_vocab.yaml \
            --plotly-font-size 9 \
            --plotly-figure-width 400 \
            --plotly-figure-height 250

        set +ex
    fi
done
