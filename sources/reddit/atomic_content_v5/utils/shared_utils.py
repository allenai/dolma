import json
import argparse
import datetime
import apache_beam as beam
from apache_beam.io import Read
from apache_beam.io.textio import WriteToText, ReadFromText
from apache_beam.io.filesystem import CompressionTypes


DATA_ACQUISITION_DATE = '2023-03-16T01:43:55.831260+00:00'


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f'Value must be positive, {value} was passed.')
    return value


def build_base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--banned_subreddits_file",
        required=False,
        help="a text file containing a list of banned subreddits.",
        default="subreddit_blocklist.txt"
    )
    parser.add_argument(
        "--input_gcs_dir",
        required=False,
        help="Input google storage directory to read the data files from."
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        help="Google cloud storage output directory to write the dataset.",
    )
    parser.add_argument(
        "--num_shards",
        default=750,
        type=positive_int,
        help="The number of shards in the dataset.",
    )
    return parser


def normalize_id(raw_id):
    """Reddit comment ids start with t1_, t2_, etc. which need to be stripped."""
    if raw_id:
        return raw_id[3:]
    return None


def normalize_string(text):
    def _safe_str(obj):
        try:
            return str(obj)
        except (UnicodeEncodeError, TypeError):
            return "[UNICODE ENCODE ERROR]"
    text = text.strip()
    return _safe_str(text)


def safe_load_json(line):
    try:
        return json.loads(line)
    except ValueError:
        return None


def isodatetime_from_epoch(epoch_time):
    try:
        return datetime.datetime.fromtimestamp(int(epoch_time), datetime.timezone.utc).isoformat()
    except TypeError:
        return epoch_time


def shuffle_posts(post_collection):
    post_collection |= "add random key" >> beam.Map(
        lambda value: (value['id'], value))
    post_collection |= "group by key" >> beam.GroupByKey()
    post_collection |= "get shuffled values" >> beam.FlatMap(lambda t: t[1])
    return post_collection


def convert_to_lm_format(example):
    text_field = 'conversational_format' if 'conversational_format' in example else 'formatted_post'

    formatted_data = {
        'text': example[text_field],
        'source': 'reddit',
        'created': example['created'],
        'added': example['added'],
        'id': example['id'],
        'metadata': {},
    }
    for k in list(example.keys()):
        if k not in [text_field] + list(formatted_data.keys()):
            formatted_data['metadata'][k] = example[k]

    formatted_data = json.dumps(formatted_data)
    return formatted_data


def load_filtered_subreddit_lists(blocked_subreddits_file):
    blocked_subreddits = [
        line.lower().strip() for line in open(
            blocked_subreddits_file,
            'r').readlines()]
    blocked_subreddits = list(set(blocked_subreddits))
    return blocked_subreddits


def read_content_from_source(content, p, args):
    if content is not None:
        content = p | ("Read in-memory content") >> beam.Create(content)
    else:
        content = p | ("Reading " + args.input_gcs_dir) >> Read(
            ReadFromText(args.input_gcs_dir)
        )
        content = content | (
            "Parse JSON" >> beam.Map(safe_load_json)
        )
    return content


def write_to_gcs(examples, banned_subreddits, args):
    examples = shuffle_posts(examples)
    examples = examples | ("Filter examples from banned subreddits" >> beam.Filter(
        lambda example: example['subreddit'].lower() not in banned_subreddits))

    file_name_suffix = ".jsonl.gz"
    serialize_fn = convert_to_lm_format
    write_sink = WriteToText
    compression_type = CompressionTypes.GZIP

    output_dir = args.output_dir
    name = 'sharded_output'
    serialized_examples = examples | (
        f'serialize {name} examples' >> beam.Map(serialize_fn))
    (
        serialized_examples | ("write " + name)
        >> write_sink(
            f'{output_dir}/{name}',
            file_name_suffix=file_name_suffix,
            compression_type=compression_type,
            max_bytes_per_shard=1000000000
        )
    )
