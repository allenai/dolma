'''
Examine how the various url taggers are excluding domains and urls

Author: Luca Soldaini (@soldni)
Email:  lucas@allenai.org
'''

from collections import Counter, defaultdict
import multiprocessing
import os
from tempfile import TemporaryDirectory
from dolma.core.paths import exists, sub_prefix
import urllib3.util
import json
from pathlib import Path
from typing import DefaultDict, Dict, List, Union
import click
import smart_open

from dolma.core.runtime import EXPERIMENT_PLACEHOLDER_NAME
from dolma.core.parallel import BaseParallelProcessor


class UPP(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(cls, queue, /, files: int = 0, documents: int = 0, matched: int = 0):
        """
        This method is called in the process_single
        to increment the progress bar.
        You can create as many progress bars as are
        the numbers of arguments after the '/' separator.
        In this example, I have created two progress
        bars, one for files and one for documents.
        The increment progressbar method should call
        the super method with the same arguments.
        """
        return super().increment_progressbar(
            queue,
            files=files,
            documents=documents,
            matched=matched,
        )

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue,
        **kwargs
    ):
        total = matched = 0
        partial = pmatched = 0
        host_counter: DefaultDict[str, int] = defaultdict(int)
        with smart_open.open(source_path, 'rt') as f:
            for ln in f:
                partial += 1
                if '"attributes":{}' in ln:
                    continue
                row = json.loads(ln)
                host = urllib3.util.parse_url(row["id"]).host
                if host is not None:
                    host_counter[host] += 1
                pmatched += 1

                if partial > 1_000:
                    cls.increment_progressbar(queue, documents=partial, matched=pmatched)
                    total += partial
                    matched += pmatched
                    partial = pmatched = 0

        cls.increment_progressbar(queue, files=1, documents=partial, matched=pmatched)
        total += partial
        matched += pmatched

        with smart_open.open(destination_path, 'wt') as f:
            doc = {'total': total, 'matched': matched, 'domains': dict(host_counter)}
            f.write(json.dumps(doc))


@click.command()
@click.option('--path', type=click.Path(file_okay=False, exists=True, path_type=Path), help='Path to tagger output', required=True)
@click.option('--top-k-domains', type=int, default=50, help='Top k domains to show')
@click.option('--processes', type=int, default=multiprocessing.cpu_count(), help='Number of workers')
@click.option('--debug', is_flag=True, help='Debug mode')
@click.option('--output', type=click.Path(path_type=Path), help='Output file', default=None)
def main(path: Path, top_k_domains: int, processes: int, debug: bool, output: Union[Path, None] = None):
    for tagger in sorted(path.iterdir()):
        if not tagger.is_dir() or tagger.name == EXPERIMENT_PLACEHOLDER_NAME:
            continue

        output_name = (output / f"{tagger.name}.json") if output else None
        if output_name is not None and exists(str(output_name)):
            print(f"Skipping {tagger.name} as it already exists")
            continue

        with TemporaryDirectory() as tmpdir:
            all_sources = [
                os.path.join(dirpath, filename)
                for dirpath, _, filenames in os.walk(tagger)
                for filename in filenames
                if filename.endswith('.gz')
            ]
            all_destinations = [
                os.path.join(tmpdir, "destination", sub_prefix(path, str(tagger)))
                for path in all_sources
            ]
            all_metadata = [
                os.path.join(tmpdir, "metadata", sub_prefix(path, str(tagger)))
                for path in all_sources
            ]

            UPP(
                source_prefix=all_sources,
                destination_prefix=all_destinations,
                metadata_prefix=all_metadata,
                num_processes=processes,
                debug=debug,
            )()

            total = matched = 0
            host_counter = Counter()
            for dirpath, _, filenames in os.walk(os.path.join(tmpdir, 'destination')):
                for filename in filenames:
                    with smart_open.open(Path(dirpath) / filename, 'rt') as f:
                        doc = json.load(f)
                        total += doc['total']
                        matched += doc['matched']
                        for host, count in doc['domains'].items():
                            host_counter[host] += count

            print(f"Tagger:  {tagger.name}")
            print(f"Total:   {total:,}")
            print(f"Matched: {matched:,}")
            print(f"Match%:  {matched / total:.2%}")
            print(f"Top {top_k_domains} domains:")
            for domain, count in host_counter.most_common(top_k_domains):
                print(f"  {domain}: {count:,}")
            print('-' * 40)

            if output_name is not None:
                output_name.parent.mkdir(parents=True, exist_ok=True)
                with open(output_name, 'w') as f:
                    json.dump(
                        {
                            "tagger": tagger.name,
                            "total": total,
                            "matched": matched,
                            "domains": host_counter.most_common(1_000),
                        },
                        f,
                        indent=4,
                    )

        # total_count = 0
        # tagger_attributes: List[dict] = []

        # # walk the tagger directory
        # for dirpath, _, filenames in os.walk(tagger):
        #     for filename in filenames:
        #         if not filename.endswith('.gz'):
        #             continue
        #         with smart_open.open(Path(dirpath) / filename, 'rt') as f:
        #             for ln in f:
        #                 total_count += 1
        #                 if '"attributes":{}' in ln:
        #                     continue
        #                 tagger_attributes.append(json.loads(ln))

        # top_domains = Counter([urllib3.util.parse_url(x["id"]).host for x in tagger_attributes])

        # print(f"Tagger:  {tagger.name}")
        # print(f"Total:   {total_count:,}")
        # print(f"Matched: {len(tagger_attributes):,}")
        # print(f"Match%:  {len(tagger_attributes) / total_count:.2%}")
        # print(f"Top {top_k_domains} domains:")
        # for domain, count in top_domains.most_common(top_k_domains):
        #     print(f"  {domain}: {count:,}")
        # print('-' * 40)


if __name__ == '__main__':
    main()
