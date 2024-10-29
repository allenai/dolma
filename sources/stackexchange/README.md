# Stack Exchange

## Instructions

1. Download the Stack Exchange data from the Internet Archive using the [`download_from_ia.sh`](download_from_ia.sh) script.
2. Convert data to parquet using the [`v0.py`](v0.py) script.
3. Load the data into Athena as follows:

Create comments table:

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS `lucas`.`se_comments_20240930` (
    Id STRING,
    PostId STRING,
    Score STRING,
    Text STRING,
    CreationDate STRING,
    UserID STRING,
    ContentLicense STRING
)
PARTITIONED BY (forum STRING)
STORED AS PARQUET
LOCATION 's3://ai2-llm/pretraining-data/sources/stackexchange/raw/20240930_parquet/comments/'
TBLPROPERTIES ('parquet.compression'='SNAPPY')
```

Then run the following to load the partitions:

```sql
MSCK REPAIR TABLE lucas.se_comments_20240930;
```

Create posts table:

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS `lucas`.`se_posts_20240930` (
    AcceptedAnswerId BIGINT,
    AnswerCount BIGINT,
    Body STRING,
    ClosedDate STRING,
    CommentCount BIGINT,
    ContentLicense STRING,
    CreationDate STRING,
    Id BIGINT,
    LastActivityDate STRING,
    LastEditDate STRING,
    LastEditorDisplayName STRING,
    LastEditorUserId BIGINT,
    OwnerDisplayName STRING,
    OwnerUserId BIGINT,
    ParentId BIGINT,
    PostTypeId STRING,
    Score BIGINT,
    Tags STRING,
    Title STRING,
    ViewCount BIGINT
)
PARTITIONED BY (forum STRING)
STORED AS PARQUET
LOCATION 's3://ai2-llm/pretraining-data/sources/stackexchange/raw/20240930_parquet/posts/'
TBLPROPERTIES ('parquet.compression'='SNAPPY')
```

Then run the following to load the partitions:

```sql
MSCK REPAIR TABLE lucas.se_posts_20240930;
```

# Selecting QA pairs


```sql
UNLOAD (
    WITH valid_questions AS (
        SELECT
            posts.Body,
            posts.Id,
            posts.CommentCount,
            posts.ContentLicense,
            posts.CreationDate,
            posts.LastActivityDate,
            posts.LastEditDate,
            posts.LastEditorDisplayName,
            posts.LastEditorUserId,
            posts.OwnerUserId,
            posts.OwnerDisplayName,
            posts.Score,
            posts.Tags,
            posts.ViewCount,
            posts.Title,
            posts.Forum,
            posts.AcceptedAnswerid
        FROM "lucas"."se_posts_20240930" as posts
        WHERE
            posttypeid = 'Question'
            AND posts.AnswerCount > 0
            AND posts.acceptedanswerid >= 0

    ),
    valid_answers AS  (
        SELECT
            posts.Body,
            posts.Id,
            posts.CommentCount,
            posts.ContentLicense,
            posts.CreationDate,
            posts.LastActivityDate,
            posts.LastEditDate,
            posts.LastEditorDisplayName,
            posts.LastEditorUserId,
            posts.OwnerUserId,
            posts.OwnerDisplayName,
            posts.Score,
            posts.ViewCount,
            posts.Forum
        FROM "lucas"."se_posts_20240930" as posts
        WHERE posttypeid = 'Answer'
    ),
    joined_questions_answers AS (
        SELECT
            valid_answers.Body AS answer_body,
            valid_answers.Id AS answer_id,
            valid_answers.CommentCount AS answer_comment_count,
            valid_answers.ContentLicense AS answer_content_license,
            valid_answers.CreationDate AS answer_creation_date,
            valid_answers.LastActivityDate AS answer_last_activity_date,
            valid_answers.LastEditDate AS answer_last_edit_date,
            valid_answers.LastEditorDisplayName AS answer_last_editor_display_name,
            valid_answers.LastEditorUserId AS answer_last_editor_user_id,
            valid_answers.OwnerUserId AS answer_owner_user_id,
            valid_answers.OwnerDisplayName AS answer_owner_display_name,
            valid_answers.Score AS answer_score,
            valid_answers.ViewCount AS answer_view_count,
            valid_answers.Forum AS answer_forum,
            valid_questions.Title AS question_title,
            valid_questions.Body AS question_body,
            valid_questions.Id AS question_id,
            valid_questions.CommentCount AS question_comment_count,
            valid_questions.ContentLicense AS question_content_license,
            valid_questions.CreationDate AS question_creation_date,
            valid_questions.LastActivityDate AS question_last_activity_date,
            valid_questions.LastEditDate AS question_last_edit_date,
            valid_questions.LastEditorDisplayName AS question_last_editor_display_name,
            valid_questions.LastEditorUserId AS question_last_editor_user_id,
            valid_questions.OwnerUserId AS question_owner_user_id,
            valid_questions.OwnerDisplayName AS question_owner_display_name,
            valid_questions.Score AS question_score,
            valid_questions.Tags AS question_tags,
            valid_questions.ViewCount AS question_view_count,
            valid_questions.Forum AS question_forum,
            CAST (
                ARRAY_MAX(
                    TRANSFORM(
                        regexp_extract_all(valid_answers.body, '\n+'),
                        x -> LENGTH(x)
                    )
                    || ARRAY [1]
                ) AS INTEGER
            ) as question_max_newline,
            CAST (
                ARRAY_MAX(
                    TRANSFORM(
                        regexp_extract_all(valid_questions.body, '\n+'),
                        x -> LENGTH(x)
                    )
                    || ARRAY [1]
                ) AS INTEGER
            ) as answer_max_newline
        FROM valid_answers
        INNER JOIN valid_questions
            ON valid_questions.forum = valid_answers.forum
            AND valid_questions.acceptedanswerid = valid_answers.id
    )
    SELECT
        (
            question_forum
            || '-'
            || CAST(question_id AS VARCHAR)
            || '-'
            || CAST(answer_id AS VARCHAR)
        ) as id,
        (
            TRIM(question_title)
            || ARRAY_JOIN(
                REPEAT(
                    CHR(10),
                    question_max_newline + 1
                ),
                ''
            )
            || TRIM(question_body)
            || ARRAY_JOIN(
                REPEAT(
                    CHR(10),
                    IF(
                        question_max_newline > answer_max_newline,
                        question_max_newline + 1,
                        answer_max_newline + 1
                    )
                ),
                ''
            )
            || TRIM(answer_body)
        ) as text,
        question_creation_date AS created,
        answer_last_activity_date AS added,
        'stackexchange' AS source,
        '20240930' as version,
        CAST(
            ROW(
                question_forum,
                question_id,
                answer_id,
                question_owner_user_id,
                answer_owner_user_id,
                question_last_editor_user_id,
                answer_last_editor_user_id,
                question_last_edit_date,
                answer_last_edit_date,
                question_last_activity_date,
                answer_last_activity_date,
                question_content_license,
                answer_content_license,
                question_score,
                answer_score,
                question_view_count,
                answer_view_count,
                question_comment_count,
                answer_comment_count
            ) AS
            ROW(
                forum VARCHAR,
                question_id BIGINT,
                answer_id BIGINT,
                question_owner_user_id BIGINT,
                answer_owner_user_id BIGINT,
                question_last_editor_user_id BIGINT,
                answer_last_editor_user_id BIGINT,
                question_last_edit_date VARCHAR,
                answer_last_edit_date VARCHAR,
                question_last_activity_date VARCHAR,
                answer_last_activity_date VARCHAR,
                question_content_license VARCHAR,
                answer_content_license VARCHAR,
                question_score BIGINT,
                answer_score BIGINT,
                question_view_count BIGINT,
                answer_view_count BIGINT,
                question_comment_count BIGINT,
                answer_comment_count BIGINT
            )
        ) AS metadata
    FROM joined_questions_answers
)
TO 's3://ai2-llm/pretraining-data/sources/stackexchange/v0/documents/20240930/'
WITH (
    format='JSON',
    compression='ZSTD'
)
```
