import csv
import hashlib
import json
import os
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
import datetime
import pandas as pd
import smart_open
import tqdm
import datasets

NEWS = {
    "9news.com.au": "hq",
    "abc.net.au": "hq",
    "abcnews.go.com": "hq",
    "afr.com": "hq",
    "aljazeera.com": "hq",
    "apnews.com": "hq",
    "bbc.com": "hq",
    "bostonglobe.com": "hq",
    "breakingnews.ie": "hq",
    "breitbart.com": "lq",
    "businessinsider.com": "hq",
    "cbc.ca": "hq",
    "cbsnews.com": "hq",
    "channel4.com": "hq",
    "chicagotribune.com": "hq",
    "cnbc.com": "hq",
    "csmonitor.com": "hq",
    "ctvnews.ca": "hq",
    "dailymail.co.uk": "lq",
    "dailystar.co.uk": "lq",
    "dw.com": "hq",
    "economist.com": "hq",
    "edition.cnn.com": "hq",
    "euronews.com": "hq",
    "express.co.uk": "hq",
    "foxnews.com": "hq",
    "france24.com": "hq",
    "globalnews.ca": "hq",
    "huffpost.com": "hq",
    "independent.co.uk": "hq",
    "independent.ie": "hq",
    "inquirer.com": "hq",
    "irishexaminer.com": "hq",
    "irishmirror.ie": "hq",
    "irishtimes.com": "hq",
    "itv.com": "hq",
    "latimes.com": "hq",
    "liverpoolecho.co.uk": "hq",
    "macleans.ca": "hq",
    "metro.co.uk": "hq",
    "mirror.co.uk": "lq",
    "montrealgazette.com": "hq",
    "morningstaronline.co.uk": "hq",
    "msnbc.com": "hq",
    "nbcnews.com": "hq",
    "news.com.au": "hq",
    "news.sky.com": "hq",
    "news.yahoo.com": "hq",
    "newshub.co.nz": "hq",
    "newsweek.com": "hq",
    "npr.org": "hq",
    "nypost.com": "lq",
    "nytimes.com": "hq",
    "nzherald.co.nz": "hq",
    "politico.com": "hq",
    "rcinet.ca": "hq",
    "reuters.com": "hq",
    "rfi.fr": "hq",
    "rnz.co.nz": "hq",
    "rt.com": "lq",
    "rte.ie": "hq",
    "sbs.com.au": "hq",
    "scoop.co.nz": "hq",
    "scotsman.com": "hq",
    "slate.com": "hq",
    "smh.com.au": "hq",
    "standard.co.uk": "hq",
    "stuff.co.nz": "hq",
    "telegraph.co.uk": "hq",
    "theage.com.au": "hq",
    "theatlantic.com": "hq",
    "theglobeandmail.com": "hq",
    "theguardian.com": "hq",
    "thehill.com": "hq",
    "thejournal.ie": "hq",
    "thestar.com": "hq",
    "thesun.co.uk": "hq",
    "thesun.ie": "hq",
    "thetimes.co.uk": "hq",
    "thewest.com.au": "hq",
    "time.com": "hq",
    "torontosun.com": "hq",
    "upi.com": "hq",
    "usatoday.com": "hq",
    "vancouversun.com": "hq",
    "walesonline.co.uk": "hq",
    "washingtonpost.com": "hq",
    "washingtontimes.com": "lq",
    "westernjournal.com": "hq",
    "wnd.com": "lq",
    "wsj.com": "hq",
}

def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


HOMEDIR = Path(os.environ['HOME'])
BASEDST = 's3://ai2-llm/pretraining-data/sources/synthetic-tiny-series/v1/documents'


def sciphi():

    sciphi_textbooks = HOMEDIR / 'tiny-series/SciPhi/textbooks-are-all-you-need-lite/data'
    sciphi_dest = f'{BASEDST}/SciPhi/textbooks-are-all-you-need-lite/part-0000.jsonl.gz'

    sciphi_data = pq.read_pandas(sciphi_textbooks).to_pandas()
    sciphi_date = datetime.datetime(2023, 9, 30)

    with smart_open.open(sciphi_dest, 'wt') as f:
        for i, row in tqdm.tqdm(
            sciphi_data.iterrows(), desc="Writing SciPhi", total=len(sciphi_data)
        ):
            row = row.to_dict()
            text = f"{row.pop('title')}\n{row.pop('completion')}".strip()
            doc = {
                "id": str(i),
                "text": text,
                "created": convert_timestamp(sciphi_date),
                "added": convert_timestamp(datetime.datetime.now()),
                "source": "SciPhi/textbooks-are-all-you-need-lite",
                "metadata": row
            }
            f.write(json.dumps(doc) + '\n')


def tiny_stories():
    tiny_stories = HOMEDIR / 'tiny-series/roneneldan/TinyStories/data'
    tiny_stories_dest = f'{BASEDST}/roneneldan/TinyStories/part-0000.jsonl.gz'

    tiny_stories_data = pq.read_table(tiny_stories).to_pandas()
    tiny_stories_date = datetime.datetime(2023, 12, 4)

    with smart_open.open(tiny_stories_dest, 'wt') as f:
        for i, row in tqdm.tqdm(
            tiny_stories_data.iterrows(), desc="Writing TinyStories", total=len(tiny_stories_data)
        ):
            # text = f"{row.pop('title')}\n{row.pop('completion')}".strip()
            doc = {
                "id": str(i),
                "text": row.text,
                "created": convert_timestamp(tiny_stories_date),
                "added": convert_timestamp(datetime.datetime.now()),
                "source": "roneneldan/TinyStories",
                "metadata": {}
            }
            f.write(json.dumps(doc) + '\n')


def nampdn_ai():
    subsets = [
        ("tiny-code-textbooks", (2024, 1, 25)),
        ("tiny-codes", (2023, 9, 29)),
        ("tiny-orca-textbooks", (2023, 9, 27)),
        ("tiny-strange-textbooks", (2024, 2, 2)),
        ("tiny-textbooks", (2024, 1, 15)),
        # ("tiny-webtext", (2023, 8, 23))
    ]
    for subset, subset_date in subsets:
        nampdn_ai = HOMEDIR / f'tiny-series/nampdn-ai/{subset}'

        if 'strange' in subset:
            partitions = [[e] for e in nampdn_ai.glob('*.parquet')]
        elif 'webtext' in subset:
            partitions = [
                list((nampdn_ai / 'train/en').glob('*.parquet')) +
                list((nampdn_ai / 'val/en').glob('*.parquet'))
            ]
        else:
            partitions = [list(nampdn_ai.glob('part-*.parquet'))]

        for i, partition in enumerate(partitions):
            nampdn_ai_dest = f'{BASEDST}/nampdn-ai/{subset}/part-{i:04d}.jsonl.gz'

            if '/train/' in str(partition):
                nampdn_ai_dest.replace(f'/{subset}/', f'/{subset}/train/')
            elif '/val/' in str(partition):
                nampdn_ai_dest.replace(f'/{subset}/', f'/{subset}/val/')

            nampdn_ai_data = pq.read_table(partition).to_pandas()
            nampdn_ai_date = datetime.datetime(*subset_date)

            if 'strange' in subset:
                iterator = ({'id': elem.Index + 1, 'text': elem.text} for elem in nampdn_ai_data.itertuples())
            else:
                iterator = (row.to_dict() for _, row in nampdn_ai_data.iterrows())

            with smart_open.open(nampdn_ai_dest, 'wt') as f:
                for row in tqdm.tqdm(
                    iterator, desc=f"Writing {subset}", total=len(nampdn_ai_data)
                ):
                    content = row.pop('response', None) or row.pop('text', None)
                    if content is None:
                        breakpoint()
                        raise ValueError(f"Could not find a response for {row}")

                    idx = row.pop('id', None) or row.pop('index', None) or row.pop('idx', None)
                    if idx is None:
                        breakpoint()
                        raise ValueError(f"Could not find an index for {row}")

                    doc = {
                        "id": str(idx),
                        "text": content.strip(),
                        "created": convert_timestamp(nampdn_ai_date),
                        "added": convert_timestamp(datetime.datetime.now()),
                        "source": f"nampdn-ai/{subset}",
                        "metadata": row
                    }
                    f.write(json.dumps(doc) + '\n')


def open_phi():
    subsets = (
        ('programming_books_llama', (2023, 10, 4)),
        ('textbooks', (2023, 10, 7))
    )

    for subset, subset_date in subsets:
        open_phi = HOMEDIR / f'tiny-series/open-phi/{subset}/data'
        open_phi_dest = f'{BASEDST}/open-phi/{subset}/part-0000.jsonl.gz'

        open_phi_data = pq.read_table(list(open_phi.glob('*.parquet'))).to_pandas()
        open_phi_date = datetime.datetime(*subset_date)

        with smart_open.open(open_phi_dest, 'wt') as f:
            for i, row in tqdm.tqdm(
                open_phi_data.iterrows(), desc=f"Writing {subset}", total=len(open_phi_data)
            ):
                row = row.to_dict()
                doc = {
                    "id": i,
                    "text": row.pop('markdown').strip(),
                    "created": convert_timestamp(open_phi_date),
                    "added": convert_timestamp(datetime.datetime.now()),
                    "source": f"open-phi/{subset}",
                    "metadata": {
                        k: list(v) if isinstance(v, np.ndarray) else v
                        for k, v in row.items()
                    }
                }
                f.write(json.dumps(doc) + '\n')


def frontpage_news_split():
    dest = "s3://ai2-llm/pretraining-data/sources/AndyReas-frontpage-news/v2/documents"
    base_path = HOMEDIR / "frontpage-news/data"

    for i, path in enumerate(base_path.glob("*.parquet")):
        data = pq.read_table(path).to_pandas()

        curr_dest_hq = f"{dest}/hq/frontpage-{i:04d}.jsonl.gz"
        curr_dest_lq = f"{dest}/lq/frontpage-{i:04d}.jsonl.gz"

        with smart_open.open(curr_dest_hq, 'wt') as hqf, smart_open.open(curr_dest_lq, 'wt') as lqf:
            for _, og_row in tqdm.tqdm(
                data.iterrows(), desc="Writing Frontpage News (split)", total=len(data)
            ):
                row = og_row.to_dict()
                year = int(row['meta']['date'][:4])
                month = int(row['meta']['date'][4:6])
                date = int(row['meta']['date'][6:8])
                text = f'{row.pop("title")}\n{row.pop("description")}'.strip()
                idx = row.pop('new_article_id')

                created = datetime.datetime(int(year), int(month), int(date))
                doc = {
                    "id": str(idx),
                    "text": text,
                    "created": convert_timestamp(created),
                    "added": convert_timestamp(datetime.datetime.now()),
                    "source": "AndyReas-frontpage-news",
                    "metadata": {"url": row['meta']['outlet'], **row.pop('meta'), **row}
                }
                if NEWS[doc['metadata']['url']] == 'hq':
                    hqf.write(json.dumps(doc) + '\n')
                else:
                    lqf.write(json.dumps(doc) + '\n')


def frontpage_news():
    dest = "s3://ai2-llm/pretraining-data/sources/AndyReas-frontpage-news/v1/documents"
    base_path = HOMEDIR / "frontpage-news/data"

    for i, path in enumerate(base_path.glob("*.parquet")):
        data = pq.read_table(path).to_pandas()
        curr_dest = f"{dest}/frontpage-{i:04d}.jsonl.gz"

        with smart_open.open(curr_dest, 'wt') as f:
            for _, og_row in tqdm.tqdm(
                data.iterrows(), desc="Writing Frontpage News", total=len(data)
            ):
                row = og_row.to_dict()
                year = int(row['meta']['date'][:4])
                month = int(row['meta']['date'][4:6])
                date = int(row['meta']['date'][6:8])
                text = f'{row.pop("title")}\n{row.pop("description")}'.strip()
                idx = row.pop('new_article_id')

                created = datetime.datetime(int(year), int(month), int(date))
                doc = {
                    "id": str(idx),
                    "text": text,
                    "created": convert_timestamp(created),
                    "added": convert_timestamp(datetime.datetime.now()),
                    "source": "AndyReas-frontpage-news",
                    "metadata": {"url": row['meta']['outlet'], **row.pop('meta'), **row}
                }
                f.write(json.dumps(doc) + '\n')


def orca_math():
    orca_math = HOMEDIR / "orca-math-word-problems-200k/data"
    orca_math_dest = (
        f"s3://ai2-llm/pretraining-data/sources/microsoft-orca-math-word-problems-200k/v1/documents/part-0000.jsonl.gz"
    )

    orca_math_data = pq.read_table(list(orca_math.glob("*.parquet"))).to_pandas()
    orca_math_date = datetime.datetime(2024, 3, 4)

    with smart_open.open(orca_math_dest, 'wt') as f:
        for i, row in tqdm.tqdm(
            orca_math_data.iterrows(), desc="Writing Orca Math", total=len(orca_math_data)
        ):
            row = row.to_dict()
            text = f"{row.pop('question')}\n{row.pop('answer')}".strip()
            doc = {
                "id": str(i),
                "text": text,
                "created": convert_timestamp(orca_math_date),
                "added": convert_timestamp(datetime.datetime.now()),
                "source": "orca-math-word-problems-200k",
                "metadata": row,
            }
            f.write(json.dumps(doc) + '\n')


def recipe_nlg():
    recipe_nlg_src = HOMEDIR / "RecipeNLG/full_dataset.csv"
    recipe_nlg_dest = f"s3://ai2-llm/pretraining-data/sources/RecipeNLG/v1/documents/part-0000.jsonl.gz"

    with smart_open.open(recipe_nlg_src, 'rt') as src_f, smart_open.open(recipe_nlg_dest, 'wt') as dst_f:

        reader = csv.DictReader(
            src_f, fieldnames=["id", "title", "ingredients", "directions", "url", "source", "entities"]
        )
        next(reader, None)  # skip the headers

        for row in tqdm.tqdm(reader, desc="Writing RecipeNLG", total=2231142):
            # text = "\n\n".join(
            #     (
            #         row.pop("title"),
            #         "\n".join(row.pop("ingredients")),
            #         "\n".join(row.pop("directions")),
            #     )
            # )
            title = row.pop("title", "").strip()
            ingredients = '\n'.join(json.loads(row.pop("ingredients", "[]")))
            directions = '\n'.join(json.loads(row.pop("directions", "[]")))

            if not title or not ingredients or not directions:
                continue
            text = f"{title}\n\n{ingredients}\n\n{directions}".strip()

            doc = {
                "id": row['id'],
                "text": text.strip(),
                "created": convert_timestamp(datetime.datetime(2020, 12, 11)),
                "added": convert_timestamp(datetime.datetime.now()),
                "source": "RecipeNLG",
                "metadata": {"url": f"http://{row['url']}", "entities": json.loads(row['entities'])}
            }
            dst_f.write(json.dumps(doc) + '\n')


def summarization():
    dest = "s3://ai2-llm/pretraining-data/sources/summarization/v1/documents"

    # cd_date = datetime.datetime(2016, 8, 26)
    # cnn_dailymail = datasets.load_dataset("ccdv/cnn_dailymail", "3.0.0")
    # for split, cn_data in cnn_dailymail.items():
    #     with smart_open.open(f"{dest}/cnn_dailymail/{split}.jsonl.gz", 'wt') as f:
    #         for row in tqdm.tqdm(cn_data, desc=f"CNN/DailyMail {split}", total=len(cn_data)):
    #             doc = {
    #                 "id": str(row['id']),
    #                 "text": row['article'].strip(),
    #                 "created": convert_timestamp(cd_date),
    #                 "added": convert_timestamp(datetime.datetime.now()),
    #                 "source": "ccdv/cnn_dailymail",
    #                 "metadata": {"summary": row['highlights']}
    #             }
    #             f.write(json.dumps(doc) + '\n')

    # xsum = datasets.load_dataset("EdinburghNLP/xsum")
    # xsum_date = datetime.datetime(2018, 8, 27)
    # for split, xs_data in xsum.items():
    #     with smart_open.open(f"{dest}/xsum/{split}.jsonl.gz", 'wt') as f:
    #         for row in tqdm.tqdm(xs_data, desc=f"XSum {split}", total=len(xs_data)):
    #             doc = {
    #                 "id": str(row['id']),
    #                 "text": row['document'].strip(),
    #                 "created": convert_timestamp(xsum_date),
    #                 "added": convert_timestamp(datetime.datetime.now()),
    #                 "source": "EdinburghNLP/xsum",
    #                 "metadata": {"summary": row['summary']}
    #             }
    #             f.write(json.dumps(doc) + '\n')

    for year in range(2017, 2024):
        for month in range(1, 13):
            if year == 2024 and month > 2:
                break

            bbc_news = datasets.load_dataset("RealTimeData/bbc_news_alltime", f"{year:04d}-{month:02d}")['train']
            with smart_open.open(f"{dest}/bbc_news/train/{year:04d}-{month:02d}.jsonl.gz", 'wt') as f:
                for row in tqdm.tqdm(bbc_news, desc=f"BBC News {year:04d}-{month:02d}", total=len(bbc_news)):
                    idx = hashlib.md5(row["link"].encode())
                    idx.update(row['published_date'].encode())

                    text = f"{row['title']}\n{row['content']}".strip()
                    a_year, a_month, a_day = row["published_date"].split('-')
                    doc = {
                        "id": idx.hexdigest(),
                        "text": text,
                        "created": convert_timestamp(datetime.datetime(int(a_year), int(a_month), int(a_day))),
                        "added": convert_timestamp(datetime.datetime.now()),
                        "source": "RealTimeData/bbc_news_alltime",
                        "metadata": {
                            "url": row["link"],
                            "authors": row["authors"],
                            "description": row["description"],
                            "section": row["section"],
                            "top_image": row["top_image"],
                            "title": row["title"],
                        },
                    }
                    f.write(json.dumps(doc) + '\n')

    multi_news = datasets.load_dataset("multi_news")
    multi_news_date = datetime.datetime(19, 6, 4)
    for split, mn_data in multi_news.items():
        with smart_open.open(f"{dest}/multi_news/{split}.jsonl.gz", 'wt') as f:
            for row in tqdm.tqdm(mn_data, desc=f"Multi News {split}", total=len(mn_data)):

                doc = {
                    "id": hashlib.md5(row['document'].encode()).hexdigest(),
                    "text": row['document'].strip(),
                    "created": convert_timestamp(multi_news_date),
                    "added": convert_timestamp(datetime.datetime.now()),
                    "source": "multi_news",
                    "metadata": {"summary": row['summary']}
                }
                f.write(json.dumps(doc) + '\n')


if __name__ == "__main__":
    # sciphi()
    # tiny_stories()
    # nampdn_ai()
    # open_phi()
    # frontpage_news()
    # frontpage_news_split()
    # orca_math()
    # recipe_nlg()
    summarization()
