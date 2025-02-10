CATEGORIES=(
    "academic_writing"
    "content_listing"
    "creative_writing"
    "customer_support_page"
    "discussion_forum_or_comment_section"
    "faqs"
    "incomplete_content"
    "knowledge_article"
    "legal_notices"
    "listicle"
    "news_article"
    "nonfiction_writing"
    "organizational_about_page"
    "organizational_announcement"
    "personal_about_page"
    "personal_blog"
    "product_page"
    "qanda_forum"
    "spam_or_ads"
    "structured_data"
    "technical_writing"
    "transcript_or_interview"
    "tutorial_or_how_to_guide"
    "user_reviews"
)

for category in "${CATEGORIES[@]}"; do
    processes=$(($(nproc) - 4))

    set -ex
    dolma tokens \
        --documents "s3://ai2-llm/pretraining-data/sources/dclm/baseline_type_classified/${category}/documents/*/*gz" \
        --destination "${HOME}/ai2-llm/preprocessed/dclm/baseline_type_classified/${category}/allenai/dolma2-tokenizer" \
        --no-tokenizer.segment_before_tokenization \
        --tokenizer.name_or_path "allenai/dolma2-tokenizer" \
        --tokenizer.eos_token_id 100257 \
        --tokenizer.pad_token_id 100277 \
	    --tokenizer.encode_special_tokens \
        --processes $(($(nproc) - 4)) \
        --seed 3920 \
        --max_size 3758096384 \
        --sample_ring_prop \
        --dtype uint32
    set +ex
done
