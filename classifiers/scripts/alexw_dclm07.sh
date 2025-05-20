#! /bin/bash

# Input parameters
DOCUMENTS='s3://ai2-llm/pretraining-data/sources/s2pdf_dedupe_minhash_v1_with_no_pii_basic_quality/documents/*.jsonl.gz'
MODEL_NAME="WebOrganizer/TopicClassifier-NoURL"
NUM_NODES=64
BATCH_SIZE=100
PRIORITY="high"
CLUSTER="ai2/augusta-google-*" #"ai2/augusta-google-*" # # "ai2/s2-*" 

# Performance tuning parameters
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
#export CUDA_LAUNCH_BLOCKING=0
#export NVIDIA_TF32_OVERRIDE=1
#export TORCH_DISTRIBUTED_DEBUG=OFF

# Generate run hash
RUN_HASH=$(echo -n "${MODEL_NAME}${DOCUMENTS}" | md5sum | awk '{print $1}')
RUN_NAME="datadelve_classifier_davidg_${RUN_HASH:0:8}"

# Resource allocation optimizations
#CPU_CORES_PER_GPU=16
#OMP_THREADS=$((CPU_CORES_PER_GPU / 2))
NUM_GPUS=1

gantry run \
    --name "${RUN_NAME}" \
    --description "Score ${DOCUMENTS} with ${MODEL_NAME}" \
    --allow-dirty \
    --workspace ai2/oe-data \
    --beaker-image 'petew/olmo-torch23-gantry' \
    --timeout -1 \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --gpus 1\
    --replicas ${NUM_NODES} \
    --preemptible \
    --cluster "${CLUSTER}" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8\
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=jakep_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=jakep_AWS_SECRET_ACCESS_KEY \
    --env-secret WANDB_API_KEY=jakep_WANDB_API_KEY \
    --env-secret HF_TOKEN=jakep_HF_TOKEN \
    --shared-memory 100GiB \
    --memory 800GiB \
    --install "pip install -e classifiers/" \
    --yes \
    -- /bin/bash -c "huggingface-cli download ${MODEL_NAME} && python \
        -m dolma_classifiers.inference \
        --source-prefix ${DOCUMENTS} \
        --batch-size ${BATCH_SIZE} \
        --use-wandb \
        --wandb-project 'dolma-classifiers' \
        --wandb-entity ai2-llm \
        --model-name ${MODEL_NAME} \
        --num-workers 1 \
        --text-key '.text'"


#        --s contrib/datacomp/DCLM-refinedweb/global_shard_01_of_10 \



# gantry run \
#     --task-name "${RUN_NAME}" \
#     --description "Score ${DOCUMENTS} with ${MODEL_NAME}" \
#     --allow-dirty \
#     --workspace ai2/davidw-oe-annealing \
#     --beaker-image 'petew/olmo-torch23-gantry' \
#     --timeout -1 \
#     --show-logs \
#     --host-networking \
#     --venv 'base' \
#     --priority "${PRIORITY}" \
#     --leader-selection \
#     --gpus 8 \
#     --replicas ${NUM_NODES} \
#     --preemptible \
#     --cluster "${CLUSTER}" \
#     --budget ai2/oe-data \
#     --env LOG_FILTER_TYPE=local_rank0_only \
#     --env OMP_NUM_THREADS=8 \
#     --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
#     --env-secret AWS_ACCESS_KEY_ID=lucas-AWS_ACCESS_KEY_ID \
#     --env-secret AWS_SECRET_ACCESS_KEY=lucas-AWS_SECRET_ACCESS_KEY \
#     --env-secret WANDB_API_KEY=lucas-WANDB_API_KEY \
#     --shared-memory 10GiB \
#     --install "pip install -e classifiers/" \
#     --yes \
#     -- /bin/bash -c "huggingface-cli download ${MODEL_NAME} && torchrun --nnodes "${NUM_NODES}:${NUM_NODES}" --nproc-per-node 8 --rdzv_id 12347 --rdzv_backend static --rdzv_endpoint "\${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" --node_rank "\${BEAKER_REPLICA_RANK}" --rdzv_conf 'read_timeout=3600' -m dolma_classifiers.inference --source-prefix ${DOCUMENTS} --batch-size ${BATCH_SIZE} --use-wandb --wandb-project 'dolma-classifiers' --wandb-entity ai2-llm --model-name ${MODEL_NAME} --num-workers 4 --text-key '.id\n.text'"
