#! /bin/bash

# DOCUMENTS='s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile/documents/*zst'
DOCUMENTS='s3://ai2-llm/pretraining-data/sources/dclm/v0/documents/40b-split/20b-01/*zstd'
# NUM_NODES=1
NUM_NODES=2
MODEL_NAME="HuggingFaceFW/fineweb-edu-classifier"
# CLUSTER="ai2/jupiter*"
CLUSTER="ai2/s2-cirrascale-l40"
# BATCH_SIZE=1024
BATCH_SIZE=128

# Generate a hash for the run name by combining model name and documents
RUN_HASH=$(echo -n "${MODEL_NAME}${DOCUMENTS}" | md5sum | awk '{print $1}')
RUN_NAME="fineweb_classifier_${RUN_HASH:0:8}"

# Set the run name as an environment variable
export BEAKER_EXPERIMENT_NAME="${RUN_NAME}"


gantry run \
    --task-name "${RUN_NAME}" \
    --description "Score ${DOCUMENTS} with ${MODEL_NAME}" \
    --allow-dirty \
    --workspace ai2/oe-data-model-based-cleanup \
    --beaker-image 'lucas/refine1' \
    --host-networking \
    --venv 'base' \
    --priority urgent \
    --leader-selection \
    --gpus 8 \
    --replicas ${NUM_NODES} \
    --preemptible \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=S2_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=S2_AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --shared-memory 10GiB \
    --install "pip install necessary s3fs wandb smart_open>=7.0.4" \
    --yes \
    -- /bin/bash -c "huggingface-cli download ${MODEL_NAME} && torchrun --nnodes "${NUM_NODES}:${NUM_NODES}" --nproc-per-node 8 --rdzv_id 12347 --rdzv_backend static --rdzv_endpoint "\${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" --node_rank "\${BEAKER_REPLICA_RANK}" --rdzv_conf 'read_timeout=420' scripts/fineweb_classifier.py --source-prefix ${DOCUMENTS} --batch-size ${BATCH_SIZE} --use-wandb --model-name ${MODEL_NAME}"
    # -- /bin/bash -c "huggingface-cli download ${MODEL_NAME} && torchrun --standalone --nproc_per_node=8 scripts/fineweb_classifier.py --source-prefix ${DOCUMENTS} --batch-size ${BATCH_SIZE} --model-name ${MODEL_NAME}"
