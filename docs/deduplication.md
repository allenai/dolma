### The `dedupe` command

The dedupe command is used to deduplicate a set of documents at the attribute level using a bloom filter.
For example configurations, see directory `tests/config`. For example:

```shell
dolma dedupe -c tests/config/dedupe-paragraphs.json
```
