# Dolma Search

Dolma Search is a toolkit for easy indexing and searching of data in Dolma format. It provides functionality to create, manage, and query indexes using the Tantivy search engine.

## Features

- Create and manage Tantivy indexes
- Index documents from various sources, including local files and S3 buckets
- Perform searches on indexed data with customizable queries
- Display search results in different formats (JSON, table, or snippet view)

## Installation

You can install Dolma Search using pip:

```shell
git clone https://github.com/allenai/dolma.git
pip install search
```

## Usage

### Indexing

To index documents, use the `dolma_search.index` module. Here's an example of how to use it:

```shell
dolma-search index \
    -i /path/to/index \
    -d "s3://ai2-llm/pretraining-data/sources/path/to/documents/*.gz"
```

The following command line options are available:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--documents` | `-d` | The documents to index. Can be any glob pattern supported by smart-open library. | Required |
| `--index-path` | `-i` | The path to the index. If not provided, an in-memory index will be used. | None |
| `--force` | `-f` | If the index already exists, delete it and create a new one. | False |
| `--num-readers` | `-n` | The number of readers to use. | 1 |
| `--num-indexers` | `-N` | The number of indexers to use. | 1 |
| `--reader-batch-size` | `-b` | The batch size for readers. | 1000 |
| `--indexer-batch-size` | `-B` | The batch size for indexers. | 1000 |
| `--heap-size` | `-H` | The heap size for the index writer. | 1GB |
| `--queue-size-per-thread` | `-q` | The size of the queue to use for storing documents. | 125 |



### Searching

To search the indexed documents, use the `dolma_search.query` module. Here's an example of how to use it:


```shell
dolma-search query \
    -i /data/flan_index \
    -q "What is the capital of France?"
```

You can also pass search queries from stdin

```shell
cat queries.txt | dolma-search query -i /data/flan_index
```

You can choose which format to display the results in. Valid options are:

- `json`: Print the results in JSON format with no coloring; perfect for piping to another program that can parse JSONL output.
- `table`: Print the results in a table format with coloring.
- `snippet`: Print the results in a table format with coloring; snippets containing matches, rather than the full document, are displayed.

Other options for the `query` command include:

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--index-path` | `-i` | The path to the index. | Required |
| `--query` | `-q` | The query to search for. If not provided, enters interactive mode. If set to "-", reads queries from stdin. | None |
| `--num-hits` | `-n` | The number of hits to return. | 10 |
| `--display-format` | `-f` | The format to display the search results in. Options: table, json, snippet. | json |
| `--selector` | `-s` | The selector used to process the queries. Uses jq syntax. | None |
