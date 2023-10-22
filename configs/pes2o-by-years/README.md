# PeS2o chronological training

## Step 1: Partitioning PeS2o V2 by year using Athena

### Step 1.1: Load current PeS2o into Athena

```sql
CREATE EXTERNAL TABLE IF NOT EXISTS `lucas`.`pes2o_v2` (
  `added` string,
  `created` string,
  `id` string,
  `metadata` string,
  `source` string,
  `version` string,
  `text` string
) COMMENT "PeS2o V2 dataset"
PARTITIONED BY (dataset string, split string, part_id int)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://ai2-llm/pretraining-data/sources/s2/v3-fos/documents/'
TBLPROPERTIES (
  'classification' = 'json',
  'write.compression' = 'GZIP'
)
```

### Step 1.2: Repair table

```sql
MSCK REPAIR TABLE pes2o_v2
```

### Step 1.3: Dump data to S3 with partitions

```sql
UNLOAD (
    SELECT
        id,
        added,
        created,
        text,
        version,
        json_format(metadata) as metadata,
        -- this should reserve about 5M tokens for validation
        IF(RANDOM() <= 0.0001, 'valid', 'train') as split,
        year
    FROM (
        SELECT
            *,
            cast(json_extract(metadata, '$.year') as int) as year
        FROM (
            SELECT
                id,
                added,
                created,
                json_parse(metadata) as metadata,
                text,
                version
            FROM "lucas"."pes2o_v2"
        )
    )
    WHERE year < 2024
)
TO 's3://ai2-llm/pretraining-data/sources/s2/v3-by-year/documents'
WITH (
    format='JSON',
    compression='GZIP',
    partitioned_by = ARRAY['split', 'year']
)
```

## Step 2: Slice Validation Data
