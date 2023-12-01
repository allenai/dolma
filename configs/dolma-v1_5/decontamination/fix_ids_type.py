import argparse
import json
from dolma.core.paths import glob_path
import tqdm

import smart_open


def fix_path(p: str):
    with smart_open.open(p, 'rt') as f:
        data = [json.loads(line) for line in f]

    with smart_open.open(p, 'wt') as f:
        for d in data:
            if 'id' in d:
                d['id'] = str(d['id'])
            f.write(json.dumps(d) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='+')
    args = ap.parse_args()

    with tqdm.tqdm(desc='Files') as pbar:
        for p in args.path:
            for sp in glob_path(p):
                fix_path(sp)
                pbar.update()


if __name__ == '__main__':
    main()
