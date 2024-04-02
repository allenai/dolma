import argparse
import json
from dolma.core.paths import glob_path
import smart_open


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--documents', type=str, required=True)
    ap.add_argument('--attributes', type=list, default=[], nargs='+')
    opts = ap.parse_args()
    return opts


def main():
    opts = parse_args()
    documents = glob_path(opts.documents)

    for path in documents:
        with smart_open.open(path, 'rt') as f:
            for row in f:
                data = json.loads(row)
                attributes = data.get('attributes', {})
                if not attributes:
                    raise ValueError('No attributes found in document')

                if opts.attributes:
                    attributes = {k: v for k, v in attributes.items() if k in opts.attributes}

                print(data["id"])
                for name, spans in attributes.items():
                    print(name)
                    for start, end, score in spans:
                        text = data["text"][start:end]
                        print(f'{score}\n{text}\n----')
                input('\n\n')


if __name__ == '__main__':
    main()
