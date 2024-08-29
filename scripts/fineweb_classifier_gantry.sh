#! /bin/bash


DOCUMENTS='s3://ai2-llm/pretraining-data/sources/dclm/v0_rep32_ft7percentile/documents/*zst'
NUM_NODES=1

gantry run \
    --description "Score DCLM7 with fineweb classifier" \
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
    --cluster "ai2/jupiter*" \
    --budget ai2/oe-data \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env BEAKER_USER_ID=$(beaker account whoami --format json | jq '.[0].name' -cr) \
    --env-secret AWS_ACCESS_KEY_ID=S2_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=S2_AWS_SECRET_ACCESS_KEY \
    --shared-memory 10GiB \
    --install "pip install necessary s3fs" \
    --yes \
    -- /bin/bash -c "torchrun --standalone --nproc_per_node=8 scripts/fineweb_classifier.py --source-prefix ${DOCUMENTS} --batch-size 1024"

# Multi-node command (commented out):
# -- /bin/bash -c "torchrun --nnodes "${NUM_NODES}:${NUM_NODES}" --nproc-per-node 8 --rdzv_id 12347 --rdzv_backend static --rdzv_endpoint "\${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" --node_rank "\${BEAKER_REPLICA_RANK}" --rdzv_conf 'read_timeout=420' scripts/fineweb_classifier.py --source-prefix ${DOCUMENTS} --batch-size 512"
