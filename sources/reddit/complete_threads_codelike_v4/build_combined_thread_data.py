"""A Dataflow script for creating DOLMA formatted pretraining data from
reddit comment and submission dumps.

For usage see README.md.

Adapted from work done at PolyAI:
https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit
"""

import logging
from functools import partial
from collections import defaultdict
import uuid
import random
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
    isodatetime_from_epoch,
    DATA_ACQUISITION_DATE
)


def parse_args(argv=None):
    parser = build_base_parser()

    parser.add_argument(
        "--parent_depth",
        type=positive_int,
        help="How many parents to consider.",
    )
    parser.add_argument(
        "--max_length",
        type=positive_int,
        default=10000,
        help="Maximum letters in content to include.",
    )
    parser.add_argument(
        "--min_length",
        type=positive_int,
        default=9,
        help="Minimum characters in content to include.",
    )
    parser.add_argument(
        "--min_conversational_length",
        type=positive_int,
        default=500,
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

    comment_score = comment.get('score', 0)
    comment = {
        'id': comment['id'],
        'thread_id': normalize_id(comment['link_id']),
        'parent_id': normalize_id(comment['parent_id']),
        'body': trim(normalize_string(comment['body']), max_length),
        'body_is_trimmed': len(comment['body']) > max_length,
        'author': normalize_string(comment['author']),
        'subreddit': normalize_string(comment['subreddit']),
        'created_utc': comment['created_utc'],
        'score': comment_score,
        'link_id': comment['link_id']
    }
    return comment

def normalize_post(post, max_length):

    body_key = "body" if "body" in post else "selftext"
    post = {
        "id": post['id'],
        "title": normalize_string(post['title']) if 'title' in post else '',
        "author": normalize_string(post['author']) if 'author' in post else '',
        "subreddit": normalize_string(post['subreddit']) if 'subreddit' in post else '',
        "subreddit_id": normalize_id(post['subreddit_id']) if 'subreddit_id' in post else '',
        "body": normalize_string(post[body_key]),
        "body_is_trimmed": len(post[body_key]) > max_length,
        "created_utc": post['created_utc'] if 'created_utc' in post else None,
        "score": post['score'] if 'score' in post else 0,
        "over_18": post['over_18'] if 'over_18' in post else True,
        "num_comments": post['num_comments'] if 'num_comments' in post else 0,
    }
    return post


def create_examples(combined_thread, parent_depth, min_length):

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
        return False

    def build_anonymous_author_names(thread):
        author_to_anonymous_name = {}
        for comment in list(thread):
            rand_int = random.randrange(1, 10 ** 4)
            random_author_name = f'user_{str(rand_int).zfill(4)}'
            author_to_anonymous_name[comment['author']] = random_author_name
        return author_to_anonymous_name

    def assemble_convo_string(cid, thread, anonymous_author_lookup, nodes, level):
        if level == parent_depth:
            return ''
        comment = thread[cid]
        convo_string = ''
        indent = ' ' * 4
        text = comment['body']
        author = anonymous_author_lookup[comment['author']]
        convo_string += f'{indent * level }#{author}#: {text}\n\n'
        for child in sorted(nodes.get(cid, []), key=lambda x: thread[x]['score'], reverse=True):
            convo_string += assemble_convo_string(child, thread, anonymous_author_lookup, nodes, level + 1)
        return convo_string

    def order_comments(thread, anonymous_author_lookup):
        nodes, roots = defaultdict(set), set()
        for cid, comment in thread.items():
            if _should_skip(comment, 1):
                continue
            if comment['link_id'].split('_')[1] == comment['parent_id']:
                roots.add(cid)
            else:
                nodes[comment['parent_id']].add(cid)
        thread_string = ''
        for cid in sorted(roots, key=lambda x: thread[x]['score'], reverse=True):
            thread_string += assemble_convo_string(cid, thread, anonymous_author_lookup, nodes, 0)
        return thread_string

    thread = combined_thread[1][0]
    thread_post = combined_thread[1][1]

    if thread and thread_post:
        thread_post = thread_post[0]
        id_to_comment = {
            comment['id']: comment for comment in list(thread)}

        anonymous_author_lookup = build_anonymous_author_names(thread)

        if thread_post:
            preamble = '#title#: ' + thread_post['title'] + '\n\n'
            if thread_post.get('body', '') and not _should_skip(thread_post, min_length):
                preamble += '#body#: ' + thread_post['body'] + '\n\n'
        else:
            preamble = ''

        conv_str = preamble + order_comments(
            id_to_comment, anonymous_author_lookup)

        example = {
            'conversational_format': conv_str,
            'subreddit': thread_post['subreddit'],
            'thread_id': thread_post['id'],
            'created': isodatetime_from_epoch(thread_post['created_utc']),
            'added': DATA_ACQUISITION_DATE,
            'id': uuid.uuid4().hex,
        }
        if len(conv_str) < 500:
            example = None
    else:
        example = None
    yield example


def run(argv=None, comments=None, posts=None):

    args, pipeline_args = parse_args(argv)
    banned_subreddits_file = args.banned_subreddits_file
    banned_subreddits = load_filtered_subreddit_lists(banned_subreddits_file)

    pipeline_options = PipelineOptions(pipeline_args, save_main_session=True,
                                       dataflow_service_options=['enable_prime'],
                                       experiments=['enable_batch_vmr',
                                                    'enable_vertical_memory_autoscaling',
                                                    'no_use_multiple_sdk_containers'])

    p = beam.Pipeline(options=pipeline_options)

    comments = read_content_from_source(comments, p, args.input_gcs_dir_comments)

    comments |= (
        "normalize content" >> beam.Map(
            partial(normalize_comment, max_length=args.max_length)))

    posts = read_content_from_source(posts, p, args.input_gcs_dir_submissions)

    posts |= (
        "normalize posts" >> beam.Map(
            partial(normalize_post, max_length=args.max_length)))


    comment_id_to_comments = comments | (
        "Key comments by thread id" >> beam.Map(
            lambda comment: (comment['thread_id'], comment)))

    post_id_to_posts = posts | (
        "Key posts by thread id" >> beam.Map(
            lambda post: (post['id'], post)))

    examples  = (
            (comment_id_to_comments, post_id_to_posts)
            | 'Join comments and submissions' >> beam.CoGroupByKey()
            | beam.FlatMap(partial(create_examples,
                    parent_depth=args.parent_depth,
                    min_length=args.min_length,
                    min_conversation_length=args.min_conversation_length
                    ))
    )

    examples = examples | ("Filtering posts that are skipped" >>
                     beam.Filter(lambda example: example is not None))

    write_to_gcs(examples, banned_subreddits, args)

    result = p.run()
    result.wait_until_finish()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    run()
