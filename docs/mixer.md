# Dolma Mixer

`dolma mix` combines data from multiple sources into a unified output. Input data must be in [dolma format](data-format.md).
Merges the named attributes and applies the configured filters. Substitutes text in any configured spans.

## Configuration

A sample configuration is available at [wikipedia-mixer.yaml](examples/wikipedia-mixer.yaml).

## Parameters

The following parameters are supported either via CLI (e.g. `dolma mix --parameter.name value`) or via config file (e.g. `dolma -c config.json mix`, where `config.json` contains `{"parameter" {"name": "value"}}`):

|Parameter|Required?|Description|
|:---:|---|---|
|`streams`|Yes| One or more streams to mix. |
|`streams[].name`|Yes| Prefix for output file name of each stream. |
|`streams[].documents`|Yes| Input document files for each stream. Accepts a single wildcard `*` character. Can be local, or an S3-compatible cloud path. |
|`streams[].attributes`|No| Merge attributes with the specified names. Looks for files by substituting `documents` with `attributes/<attribute_name>` in the path of each input document file. |
|`streams[].output.path`|Yes| Output will be uploaded to the S3 `path`.|
|`streams[].output.max_size_in_bytes`|No| Data will be coalesced into files no bigger than `max_size_in_bytes`. |
|`streams[].output.discard_fields`|No| Top-level fields in the `discard_fields` list will be dropped from the output documents. |
|`streams[].filter.include`|No| Optional content-based filtering. Default = keep everything. Documents are retained if they match any of the `include` patterns (or if no `include` patterns are specified) AND if they match none of the `exclude` patterns. Pattern syntax is [jsonpath](https://support.smartbear.com/alertsite/docs/monitors/api/endpoint/jsonpath.html#filters). |
|`streams[].filter.exclude`|No| Optional content-based filtering. Default = keep everything. Documents are retained if they match any of the `include` patterns (or if no `include` patterns are specified) AND if they match none of the `exclude` patterns. Pattern syntax is [jsonpath](https://support.smartbear.com/alertsite/docs/monitors/api/endpoint/jsonpath.html#filters). |
|`streams[].span_replacement`|No| A list of objects specifying spans of text to be replaced. |
|`streams[].span_replacement[].span`|No| A json-path expression for an attribute that contains an array of spans. Each span should be list of length three:  `[start, end, score]`. |
|`streams[].span_replacement[].min_score`|No| If the span score is less than this value, the span will not be replaced. |
|`streams[].span_replacement[].replacement`|No| The text that should be inserted in place of the span. Use `{}` to represent the original text. |
|`work_dir.input`|No| Path to a local scratch directory where temporary input files can be placed. If not provided, Dolma will make one for you and delete it upon completion. |
|`work_dir.output`|No| Path to a local scratch directory where temporary output files can be placed. If not provided, Dolma will make one for you and delete it upon completion. |
|`processes`|No| Number of processes to use for mixing. By default 1 process is used. |
|`dryrun`|No| If true, only print the configuration and exit without running the mixer. |
