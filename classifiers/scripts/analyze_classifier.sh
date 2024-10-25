#! /bin/bash

PRIORITY="high"
RUN_NAME=$1
shift
ARGS=$@


gantry run \
    --task-name "train_classifier" \
    --description "$RUN_NAME" \
    --allow-dirty \
    --workspace ai2/cheap_decisions \
    --beaker-image 'petew/olmo-torch23-gantry' \
    --timeout 0 \
    --host-networking \
    --venv 'base' \
    --priority "${PRIORITY}" \
    --gpus 1 \
    --preemptible \
    --cluster ai2/jupiter-cirrascale-2 \
    --cluster ai2/pluto-cirrascale \
    --cluster ai2/saturn-cirrascale \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env-secret WANDB_API_KEY=BENB_WANDB_API_KEY \
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
    --install "pip install -e classifiers/" \
    --yes \
    -- /bin/bash -c "python -m dolma_classifiers.analyze $ARGS"
