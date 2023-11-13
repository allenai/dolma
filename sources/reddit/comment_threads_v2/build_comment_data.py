"""A Dataflow script for creating DOLMA formatted pretraining data from
reddit comment dumps.

For usage see README.md.

Adapted from work done at PolyAI:
https://github.com/PolyAI-LDN/conversational-datasets/tree/master/reddit
"""


import logging
from functools import partial
import uuid
import random
from collections import defaultdict
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
        default=12,
        help="How many parents to consider.",
    )
    parser.add_argument(
        "--max_length",
        type=positive_int,
        default=2048,  # was 127 then 512.
        help="Maximum letters in content to include.",
    )
    parser.add_argument(
        "--min_length",
        type=positive_int,
        default=9,
        help="Minimum letters in content to include.",
    )
    return parser.parse_known_args(argv)


def normalize_comment(comment, max_length):

    def trim(text, max_length):
        """Trims text to be at most `max_length`, without splitting apart words."""
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


def create_examples(thread, parent_depth, min_length):
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

    def linear_paths(id_to_comment, parent_depth):
        """Gets all linear paths of content and replies from the thread.
        Each linear path is guaranteed to contain at least two comments."""
        paths = []
        seen_ids = set()
        id_to_children = defaultdict(list)
        for comment_id, comment in list(id_to_comment.items()):
            id_to_children[comment['parent_id']].append(comment_id)
            if comment['parent_id'] not in id_to_comment:
                paths.append([comment_id])
                seen_ids.add(comment_id)

        # paths start with root content
        while paths:
            new_paths = []
            for path in paths:
                last_id = path[-1]
                for child_id in id_to_children[last_id]:
                    if child_id in seen_ids:
                        # Prevent infinite loops.
                        continue
                    seen_ids.add(child_id)
                    new_path = path[-parent_depth:] + [child_id]
                    new_paths.append(new_path)

            # yield all unique paths at the tallest depth
            if len(paths) > 0:
                for path in paths:
                    root_node = path[0]
                    last_node = path[-1]
                    if root_node not in [p[0] for p in new_paths]:
                        yield path
                    elif last_node not in [p[-2] for p in new_paths]:
                        yield path
            paths = new_paths

    def trim_paths(linear_paths, id_to_comment, min_length):
        clean_paths = []
        for p in linear_paths:
            clean_path = []
            for c in p:
                if not _should_skip(id_to_comment[c], min_length):
                    clean_path.append(c)
                else:
                    break
            if len(clean_path) >= 2:
                clean_paths.append(clean_path)
        return clean_paths

    def choose_longest_unique_path(linear_paths):
        longest_paths = []
        seen_comments = []
        for p in linear_paths[::-1]:
            if p[0] not in seen_comments:
                longest_paths.append(p)
                seen_comments.extend(p)
        return longest_paths

    def build_anonymous_author_names(thread):
        author_to_anonymous_name = {}
        for comment in list(thread):
            rand_int = random.randrange(1, 10 ** 4)
            random_author_name = f'user_{str(rand_int).zfill(4)}'
            author_to_anonymous_name[comment['author']] = random_author_name
        return author_to_anonymous_name

    id_to_comment = {
        comment['id']: comment for comment in list(thread)}

    anonymous_author_lookup = build_anonymous_author_names(thread)

    all_linear_paths = [p for p in linear_paths(id_to_comment, parent_depth) if len(p) >= 2]
    cleaned_paths = trim_paths(all_linear_paths, id_to_comment, min_length)
    longest_linear_paths = choose_longest_unique_path(cleaned_paths)

    for linear_path in longest_linear_paths:
        response = id_to_comment[linear_path[-1]]
        context = id_to_comment[linear_path[-2]]  # guaranteed to exist.

        example = {
            'subreddit': response['subreddit'],
            'thread_id': response['thread_id'],
            'created': isodatetime_from_epoch(response['created_utc']),
            'added': DATA_ACQUISITION_DATE,
            'id': uuid.uuid4().hex,
        }

        def _get_author_offsets(author, conv_string):
            return len(conv_string), len(author) + len(conv_string)

        conv_str = ''
        author_offsets = []
        comment_scores = []
        for i, parent_id in enumerate(linear_path[:-2]):
            if i >= parent_depth:
                break
            author = id_to_comment[parent_id]['author']
            author_offsets.append(_get_author_offsets(author, conv_str))
            comment_scores.append(id_to_comment[parent_id]['score'])
            conv_str += f"#{anonymous_author_lookup[author]}#: {id_to_comment[parent_id]['body']}\n\n"
        author_offsets.append(_get_author_offsets(context['author'], conv_str))
        conv_str += f"#{anonymous_author_lookup[context['author']]}#: {context['body']}"
        author_offsets.append(_get_author_offsets(response['author'], conv_str))
        conv_str += f"#{anonymous_author_lookup[response['author']]}#: {response['body']}"
        comment_scores.append(context['score'])
        comment_scores.append(response['score'])

        example['conversational_format'] = conv_str
        example['user_id_spans'] = author_offsets
        example['comment_scores'] = comment_scores

        yield example


def run(argv=None, comments=None):

    args, pipeline_args = parse_args(argv)
    banned_subreddits_file = args.banned_subreddits_file
    banned_subreddits = load_filtered_subreddit_lists(banned_subreddits_file)

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

    thread_id_to_comments = comments | (
        "Key by thread id" >> beam.Map(
            lambda comment: (comment['thread_id'], comment)))

    threads = thread_id_to_comments | (
        "Group content by thread ID" >> beam.GroupByKey())

    threads = threads | ("Get threads" >> beam.Map(lambda t: t[1]))

    examples = threads | (
        "Create examples" >> beam.FlatMap(
            partial(create_examples,
                    parent_depth=args.parent_depth,
                    min_length=args.min_length,
                    )))

    # examples = shuffle_posts(examples)

    write_to_gcs(examples, banned_subreddits, args)

    result = p.run()
    result.wait_until_finish()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    run()
