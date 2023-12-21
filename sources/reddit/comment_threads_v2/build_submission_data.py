"""A Dataflow script for creating DOLMA formatted pretraining data from
reddit submission dumps.

For usage see README.md.

Adapted from work done at PolyAI:
https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit
"""

import logging
from functools import partial
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from utils.shared_utils import (
    build_base_parser,
    positive_int,
    normalize_id,
    normalize_string,
    load_filtered_subreddit_lists,
    read_content_from_source,
    write_to_gcs,
    DATA_ACQUISITION_DATE
)

def parse_args(argv=None):
    parser = build_base_parser()

    parser.add_argument(
        "--max_length",
        type=positive_int,
        default=10000,
        help="Maximum words in posts to include.",
    )
    parser.add_argument(
        "--min_length",
        type=positive_int,
        default=60,
        help="Minimum words in posts to include.",
    )
    parser.add_argument(
        "--min_score",
        default=2,
        type=positive_int,
        help="The minimum score of a post to include.",
    )
    return parser.parse_known_args(argv)


def normalize_post(post, max_length):

    body_key = "body" if "body" in post else "selftext"
    post = {
        "id": post['id'],
        "title": normalize_string(post['title']) if 'title' in post else None,
        "author": normalize_string(post['author']) if 'author' in post else None,
        "subreddit": normalize_string(post['subreddit']) if 'subreddit' in post else None,
        "subreddit_id": normalize_id(post['subreddit_id']) if 'subreddit_id' in post else None,
        "body": normalize_string(post[body_key]),
        "body_is_trimmed": len(post[body_key]) > max_length,
        "created_utc": post['created_utc'] if 'created_utc' in post else None,
        "score": post['score'] if 'score' in post else None,
        "over_18": post['over_18'] if 'over_18' in post else None,
        "num_comments": post['num_comments'] if 'num_comments' in post else None,
    }
    return post


def create_examples(post, min_length, min_score):

    def _should_skip(post, min_length, min_score):

        for key in post:
            if post[key] is None:
                return True

        if post['body_is_trimmed']:
            return True
        if post['body'] in {
                "[deleted]",
                "[removed]",
                "[UNICODE ENCODE ERROR]"} or post['author'] in {
                "[deleted]",
                "[removed]",
                "[UNICODE ENCODE ERROR]"}:
            return True
        if post['subreddit'] in {
            "[deleted]",
            "[removed]",
                "[UNICODE ENCODE ERROR]"}:
            return True
        if len(post['body']) < min_length or post['over_18'] or post['score'] < min_score:
            return True
        return False

    if _should_skip(post, min_length, min_score):
        return None

    example = {
        "id": post['id'],
        "title": post['title'],
        "author": post['author'],
        "subreddit": post['subreddit'],
        "body": post['body'],
        "score": post['score'],
        "over_18": post['over_18'],
        "num_comments": post['num_comments'],
        'added': DATA_ACQUISITION_DATE

    }
    if post['created_utc'] is None:
        example["created"] = None
    else:
        try:
            example["created"] = int(post['created_utc'])
        except ValueError:
            example["created"] = None

    example["created"] = post['created_utc']

    formatted_post_str = ''
    if example["title"] is not None:
        formatted_post_str += f"Title: {example['title']}\n\n\n"
    formatted_post_str += example['body']
    example['formatted_post'] = formatted_post_str

    yield example


def run(argv=None, posts=None):

    args, pipeline_args = parse_args(argv)
    banned_subreddits_file = args.banned_subreddits_file
    banned_subreddits = load_filtered_subreddit_lists(banned_subreddits_file)

    pipeline_options = PipelineOptions(pipeline_args, save_main_session=True)
    p = beam.Pipeline(options=pipeline_options)

    posts = read_content_from_source(posts, p, args)

    posts |= (
        "normalize posts" >> beam.Map(
            partial(normalize_post, max_length=args.max_length)))

    posts = posts | ("Filtering posts that are skipped" >>
                     beam.Filter(lambda post: post is not None))

    examples = posts | (
        "Create examples" >> beam.FlatMap(
            partial(create_examples,
                    min_length=args.min_length,
                    min_score=args.min_score,
                    )))

    write_to_gcs(examples, banned_subreddits, args)

    result = p.run()
    result.wait_until_finish()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    run()
