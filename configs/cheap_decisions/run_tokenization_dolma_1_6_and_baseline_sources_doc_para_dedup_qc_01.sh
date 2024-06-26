set -Eeuo pipefail

# dolma -c configs/cheap_decisions/dolma_cc_head_dedup_qc_01_retokenize.yaml tokens
# dolma -c configs/cheap_decisions/dolma_cc_middle_dedup_qc_01_retokenize.yaml tokens
# dolma -c configs/cheap_decisions/dolma_cc_tail_dedup_qc_01_retokenize.yaml tokens
# dolma -c configs/cheap_decisions/c4_docpara_dedup_qc_01_retokenize.yaml tokens
dolma -c configs/cheap_decisions/falcon-refinedweb-qc-plus-retokenize.yaml tokens