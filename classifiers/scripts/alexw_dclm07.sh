#! /bin/bash

DOCUMENTS='s3://ai2-llm/pretraining-data/sources/dclm/v0/documents/smol-test/*' #40b-split/20b-01/0003_dclm_shard_0001*zstd' #'s3://ai2-llm/pretraining-data/sources/dclm/v0/documents/100b/0000_dclm_shard_0000*zstd'
#DOCUMENTS='s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile/documents/*zst'

NUM_NODES=1
MODEL_NAME="data-delve/gte-base-en-v1.5_topic-v3.8_url1" #"HuggingFaceFW/fineweb-edu-classifier" 
CLUSTER="ai2/neptune-cirrascale"
BATCH_SIZE=1024
PRIORITY="high"

# Test Values
# DOCUMENTS='s3://ai2-llm/pretraining-data/sources/dclm/v0/documents/40b-split/20b-01/*zstd'
# NUM_NODES=1
# BATCH_SIZE=1024
# CLUSTER="ai2/neptune*"
# PRIORITY="high"

# Generate a hash for the run name by combining model name and documents
RUN_HASH=$(echo -n "${MODEL_NAME}${DOCUMENTS}" | md5sum | awk '{print $1}')
RUN_NAME="alexw_classifier_davidg_${RUN_HASH:0:8}"

# Set the run name as an environment variable
export BEAKER_EXPERIMENT_NAME="${RUN_NAME}"


gantry run \
    --task-name "${RUN_NAME}" \
    --description "Score ${DOCUMENTS} with ${MODEL_NAME}" \
    --allow-dirty \
    --workspace ai2/oe-data \
    --beaker-image 'petew/olmo-torch23-gantry' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 8 \
    --replicas ${NUM_NODES} \
    --preemptible \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=davidg_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=davidg_AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=davidg_WANDB_API_KEY \
    --env-secret HF_TOKEN=davidg_HF_TOKEN \
    --shared-memory 10GiB \
    --install "pip install -e classifiers/" \
    --yes \
    -- /bin/bash -c "huggingface-cli download ${MODEL_NAME} && torchrun --nnodes "${NUM_NODES}:${NUM_NODES}" --nproc-per-node 8 --rdzv_id 12347 --rdzv_backend static --rdzv_endpoint "\${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" --node_rank "\${BEAKER_REPLICA_RANK}" --rdzv_conf 'read_timeout=420' -m dolma_classifiers.inference --source-prefix ${DOCUMENTS} --batch-size ${BATCH_SIZE} --use-wandb --wandb-project 'dolma-classifiers' --wandb-entity ai2-llm --model-name ${MODEL_NAME} --num-workers 4"
