"""A Dataflow script for creating DOLMA formatted pretraining data from
reddit comment dumps.

For usage see README.md.

Adapted from work done at PolyAI:
https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit
"""

import logging
from functools import partial
import uuid
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from utils.shared_utils import (
    read_content_from_source,
    build_base_parser,
    positive_int,
    normalize_id,
    normalize_string,
    load_filtered_subreddit_lists,
    write_to_gcs,
    isodatetime_from_epoch,
    DATA_ACQUISITION_DATE
)


def parse_args(argv=None):
    parser = build_base_parser()

    parser.add_argument(
        "--parent_depth",
        type=positive_int,
        default=12,
        help="How many parent content to consider.",
    )
    parser.add_argument(
        "--max_length",
        type=positive_int,
        default=2048,
        help="Maximum letters in content to include.",
    )
    parser.add_argument(
        "--min_length",
        type=positive_int,
        default=400,
        help="Minimum characters in content to include.",
    )
    return parser.parse_known_args(argv)


def normalize_comment(comment, max_length):

    def trim(text, max_length):
        if len(text) <= max_length:
            return text
        text = text[:max_length + 1]

        # Trim until the last two characters are the boundary between an
        # alphanumeric character, and a non-alphanumeric character.
        while len(text) > 1 and (text[-1].isalnum() == text[-2].isalnum()):
            text = text[:-1]
        return text[:-1]

    comment = {
        'id': comment['id'],
        'thread_id': normalize_id(comment['link_id']),
        'parent_id': normalize_id(comment['parent_id']),
        'body': trim(normalize_string(comment['body']), max_length),
        'body_is_trimmed': len(comment['body']) > max_length,
        'author': normalize_string(comment['author']),
        'subreddit': normalize_string(comment['subreddit']),
        'created_utc': comment['created_utc'],
        'score': comment['score'],
        'link_id': comment['link_id']
    }
    return comment


def create_examples(comment, min_length):
    def _should_skip(comment, min_length):
        if comment['body_is_trimmed']:
            return True
        if comment['body'] in {
            "[deleted]",
            "[removed]",
            "[UNICODE ENCODE ERROR]"}:
            return True
        if comment['subreddit'] in {
            "[deleted]",
            "[removed]",
            "[UNICODE ENCODE ERROR]"}:
            return True
        if len(comment['body']) < min_length:
            return True
        if not comment['score'] or comment['score'] < 2:
            return True
        return False

    if _should_skip(comment, min_length):
        return None

    example = {
        'subreddit': comment['subreddit'],
        'thread_id': comment['thread_id'],
        'created': isodatetime_from_epoch(comment['created_utc']),
        'added': DATA_ACQUISITION_DATE,
        'id': uuid.uuid4().hex,
        'conversational_format': comment['body']
    }
    yield example


def run(argv=None, comments=None):
    args, pipeline_args = parse_args(argv)
    banned_subreddits_file = args.banned_subreddits_file
    banned_subreddits = load_filtered_subreddit_lists(banned_subreddits_file)
    # pipeline_options = PipelineOptions(pipeline_args, save_main_session=True, min_ram="8GB")
    # pipeline_options = PipelineOptions(pipeline_args, save_main_session=True)

    pipeline_options = PipelineOptions(pipeline_args, save_main_session=True,
                                       dataflow_service_options=['enable_prime'],
                                       experiments=['enable_batch_vmr',
                                                    'enable_vertical_memory_autoscaling',
                                                    'no_use_multiple_sdk_containers'])

    p = beam.Pipeline(options=pipeline_options)

    comments = read_content_from_source(comments, p, args)

    comments |= (
        "normalize content" >> beam.Map(
            partial(normalize_comment, max_length=args.max_length)))

    examples = comments | (
        "Create examples" >> beam.FlatMap(
            partial(create_examples,
                    min_length=args.min_length,
                    )))

    examples = examples | (
            "Filter none content" >> beam.Filter(
        lambda c: c is not None))

    write_to_gcs(examples, banned_subreddits, args)

    result = p.run()
    result.wait_until_finish()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    run()
