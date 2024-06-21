set -Eeuo pipefail

dolma -c configs/cheap_decisions/c4_docpara_dedup_retokenize.yaml tokens
dolma -c configs/cheap_decisions/dolma_cc_head_docpara_dedup_retokenize.yaml tokens
dolma -c configs/cheap_decisions/dolma_cc_middle_docpara_dedup_retokenize.yaml tokens
dolma -c configs/cheap_decisions/dolma_cc_tail_docpara_dedup_retokenize.yaml tokens