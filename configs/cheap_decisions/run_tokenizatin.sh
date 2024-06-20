set -Eeuo pipefail

dolma -c configs/cheap_decisions/books_retokenize.yaml tokens
dolma -c configs/cheap_decisions/falcon-refinedweb_retokenize.yaml tokens
dolma -c configs/cheap_decisions/reddit_retokenize.yaml tokens

dolma -c configs/cheap_decisions/wiki_retokenize.yaml tokens
dolma -c configs/cheap_decisions/redpajama_stackexchange_retokenize.yaml tokens
dolma -c configs/cheap_decisions/redpajama_arxiv_retokenize.yaml tokens
dolma -c configs/cheap_decisions/proof-pile-2-openwebmath_retokenize.yaml tokens
dolma -c configs/cheap_decisions/proof-pile-2-algebraic-stack_retokenize.yaml tokens

dolma -c configs/cheap_decisions/tulu_flan_retokenize.yaml tokens
dolma -c configs/cheap_decisions/starcoder_retokenize.yaml tokens
dolma -c configs/cheap_decisions/c4_retokenize.yaml tokens
dolma -c configs/cheap_decisions/dolma_cc_head_retokenize.yaml tokens
dolma -c configs/cheap_decisions/dolma_cc_middle_retokenize.yaml tokens
dolma -c configs/cheap_decisions/dolma_cc_tail_retokenize.yaml tokens

dolma -c configs/cheap_decisions/pes2o_retokenize.yaml tokens