'''
Examine how the various url taggers are excluding domains and urls

Author: Luca Soldaini (@soldni)
Email:  lucas@allenai.org
'''

from collections import Counter
import urllib3.util
import json
from pathlib import Path
from typing import List
import click
import smart_open


@click.command()
@click.option('--path', type=click.Path(file_okay=False, exists=True, path_type=Path), help='Path to tagger output', required=True)
@click.option('--top-k-domains', type=int, default=50, help='Top k domains to show')
def main(path: Path, top_k_domains: int):
    for tagger in sorted(path.iterdir()):
        if not tagger.is_dir():
            continue

        total_count = 0
        tagger_attributes: List[dict] = []

        # walk the tagger directory
        for dirpath, _, filenames in tagger.walk():
            for filename in filenames:
                if not filename.endswith('.gz'):
                    continue
                with smart_open.open(dirpath / filename, 'rt') as f:
                    for ln in f:
                        total_count += 1
                        if '"attributes":{}' in ln:
                            continue
                        tagger_attributes.append(json.loads(ln))

        top_domains = Counter([urllib3.util.parse_url(x["id"]).host for x in tagger_attributes])

        print(f"Tagger:  {tagger.name}")
        print(f"Total:   {total_count:,}")
        print(f"Matched: {len(tagger_attributes):,}")
        print(f"Match%:  {len(tagger_attributes) / total_count:.2%}")
        print(f"Top {top_k_domains} domains:")
        for domain, count in top_domains.most_common(top_k_domains):
            print(f"  {domain}: {count:,}")
        print('-' * 40)


if __name__ == '__main__':
    main()
