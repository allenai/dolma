#! /bin/bash

 LIST=(
     "gutenberg_books"
     "pes20_stem_papers"
     "wikipedia_wikibooks"
     "megawika"
     "stackexchange"
     "arxiv"
     "algebraic_stack"
     "openwebmath"
     "tulu"
     "cc_news"
     "starcoder"
     "c4"
     "reddit"
     "falcon"
     "web_rest"
     "all_red_pajama"
     "cc_eli5_oh_top10p"
     "falcon_eli5_oh_top10p"
     "cc_eli5_oh_top20p"
     "falcon_eli5_oh_top20p"
     "cc_og_eli5_oh_top10p"
     "falcon_og_eli5_oh_top10p"
     "prox_fineweb_pro"
     "fineweb_edu_dedup"
     "cc_tulu_qc_top10"
     "falcon_tulu_qc_top10"
     "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p"
     "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top20p"
     "regression_synthetic_20epochs_bs640_lf1_lre35_top10p"
     "regression_synthetic_20epochs_bs640_lf1_lre35_top20p"
     "dclm_ft7percentile_fw2"
     "dclm_ft7percentile_fw3"
     "dclm_fw_top3"
     "dclm_fw_top10"
     "web_instruct"
#     "DCLM-baseline"
 )

 # bash it all
  for i in "${LIST[@]}"
  do
      bash classifiers/scripts/corpus_eval/beaker/$i.sh
  done