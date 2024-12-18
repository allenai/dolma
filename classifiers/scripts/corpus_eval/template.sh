#! /bin/bash

CORPUS_NAME="{{corpus_name}}"
DOCUMENTS="{{documents_path}}"


MAX_ROWS={{max_rows}}
NUM_NODES={{num_nodes}}
NUM_GPUS={{num_gpus}}
MAX_ROWS_PER_NODE=$((MAX_ROWS / NUM_NODES))
MODEL_NAME={{model_name}}
BATCH_SIZE=512
PRIORITY="high"
# PRIORITY="urgent"
OUTPUT_PREFIX="{{output_path}}"

# Generate a hash for the run name by combining model name and documents
RUN_HASH=$(echo -n "${MODEL_NAME}${DOCUMENTS}" | md5sum | awk '{print $1}')
RUN_NAME="synthetic_mmlu_${RUN_HASH:0:8}"

# Set the run name as an environment variable
export BEAKER_EXPERIMENT_NAME="${RUN_NAME}"


gantry run \
    --task-name "${RUN_NAME}" \
    --description "Score ${DOCUMENTS} with ${MODEL_NAME}" \
    --allow-dirty \
    --workspace ai2/cheap_decisions \
    --beaker-image 'petew/olmo-torch23-gantry' \
    --show-logs \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --leader-selection \
    --gpus 1 \
    --replicas ${NUM_NODES} \
    --preemptible \
    --cluster "ai2/jupiter*" \
    --cluster "ai2/saturn-cirrascale" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret WANDB_API_KEY=BENB_WANDB_API_KEY \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
    --shared-memory 10GiB \
    --install "pip install -e classifiers/" \
    --yes \
    -- /bin/bash -c "python -m dolma_classifiers.inference --source-prefix ${DOCUMENTS} --batch-size ${BATCH_SIZE} --model-name ${MODEL_NAME} --num-workers 4 --max-length 512 --model-compile --max-rows ${MAX_ROWS_PER_NODE} --output-prefix ${OUTPUT_PREFIX} --override --max-files 150"