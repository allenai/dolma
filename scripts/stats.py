import argparse
import multiprocessing
import os
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, Tuple, Type, TypeVar, Union

import tldextract
import blingfire

from dolma.core.parallel import BaseParallelProcessor, QueueType
from dolma.core.data_types import InputSpec, OutputSpec
import msgspec
import smart_open


T = TypeVar('T', bound=Type['BaseStatsProcessor'])

PRONOUNS = (
    ('she', 'her', 'her', 'hers', 'herself'),
    ('he', 'him', 'his', 'his', 'himself'),
    ('they', 'them', 'their', 'theirs', 'themselves'),
    ('ze', 'hir', 'hir', 'hirs', 'hirself'),
    ('ze', 'zir', 'zir', 'zirs', 'zirself'),
    ('xey', 'xem', 'xyr', 'xyrs', 'xemself'),
    ('ae', 'aer', 'aer', 'aers', 'aerself'),
    ('e', 'em', 'eir', 'eirs', 'emself'),
    ('ey', 'em', 'eir', 'eirs', 'eirself'),
    ('fae', 'faer', 'faer', 'faers', 'faerself'),
    ('fey', 'fem', 'feir', 'feirs', 'feirself'),
    ('hu', 'hum', 'hus', 'hus', 'humself'),
    ('it', 'it', 'its', 'its', 'itself'),
    ('jee', 'jem', 'jeir', 'jeirs', 'jemself'),
    ('kit', 'kit', 'kits', 'kits', 'kitself'),
    ('ne', 'nem', 'nir', 'nirs', 'nemself'),
    ('peh', 'pehm', 'peh\'s', 'peh\'s', 'pehself'),
    ('per', 'per', 'per', 'pers', 'perself'),
    ('sie', 'hir', 'hir', 'hirs', 'hirself'),
    ('se', 'sim', 'ser', 'sers', 'serself'),
    ('shi', 'hir', 'hir', 'hirs', 'hirself'),
    ('si', 'hyr', 'hyr', 'hyrs', 'hyrself'),
    ('they', 'them', 'their', 'theirs', 'themself'),
    ('thon', 'thon', 'thons', 'thons', 'thonself'),
    ('ve', 'ver', 'vis', 'vis', 'verself'),
    ('ve', 'vem', 'vir', 'virs', 'vemself'),
    ('vi', 'ver', 'ver', 'vers', 'verself'),
    ('vi', 'vim', 'vir', 'virs', 'vimself'),
    ('vi', 'vim', 'vim', 'vims', 'vimself'),
    ('xie', 'xer', 'xer', 'xers', 'xerself'),
    ('xe', 'xem', 'xyr', 'xyrs', 'xemself'),
    ('xey', 'xem', 'xeir', 'xeirs', 'xemself'),
    ('yo', 'yo', 'yos', 'yos', 'yosself'),
    ('ze', 'zem', 'zes', 'zes', 'zirself'),
    ('ze', 'mer', 'zer', 'zers', 'zemself'),
    ('zee', 'zed', 'zeta', 'zetas', 'zedself'),
    ('zie', 'zir', 'zir', 'zirs', 'zirself'),
    ('zie', 'zem', 'zes', 'zes', 'zirself'),
    ('zie', 'hir', 'hir', 'hirs', 'hirself'),
    ('zme', 'zmyr', 'zmyr', 'zmyrs', 'zmyrself'),
)


class Registry:
    __registry__: Dict[str, Type['BaseStatsProcessor']] = {}

    @classmethod
    def add(cls, obj: T) -> T:
        cls.__registry__[obj.__name__] = obj
        return obj

    @classmethod
    def get(cls, name: str) -> Type['BaseStatsProcessor']:
        return cls.__registry__[name]

    @classmethod
    def all(cls) -> Generator[Tuple[str, Type['BaseStatsProcessor']], None, None]:
        yield from cls.__registry__.items()


class BaseStatsProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(  # type: ignore
        cls,
        queue: QueueType,  # queue must be the first argument, and it should be a positional-only argument
        /,
        files: int = 0,
        documents: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def cli(
        cls,
        num_workers: int = multiprocessing.cpu_count(),
        debug: bool = False,
        **process_single_kwargs: Any
    ) -> None:
        raise NotImplementedError()


@Registry.add
class common_crawl(BaseStatsProcessor):
    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue[Union[Tuple[int, ...], None]],
        **kwargs: Any
    ):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # for the data sheet, what statistics you think we should include? I could do # of docs, # tokens, distribution of URLs, pronouns, s2 FOS, stack languages?
        decoder = msgspec.json.Decoder(InputSpec)

        domains: Dict[str, int] = {}
        documents_count = flush_counter = 0
        tokens_count = 0
        pronouns: Dict[str, int] = dict((prs, 0) for pr in PRONOUNS for prs in pr)

        interval = 10_000
        max_dict_size = 100_000
        flush_every = 250_000

        with smart_open.open(source_path, "rb") as source_file:
            for line in source_file:
                document = decoder.decode(line)
                url = tldextract.extract(document.id)
                domain = '.'.join(url[1:])

                documents_count += 1

                if domain in domains:
                    domains[domain] += 1
                elif len(domains) < max_dict_size:
                    domains[domain] = 1
                elif flush_counter >= flush_every:
                    # remove all elements in domains that have <= 1 occurrence
                    domains = {k: v for k, v in domains.items() if v > 1}
                    flush_counter = 0
                else:
                    flush_counter += 1

                for w in blingfire.text_to_words(document.text).split():
                    tokens_count += 1
                    if (w := w.lower()) in pronouns:
                        pronouns[w] += 1

                if documents_count % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=documents_count % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            content = {
                'domains': domains, 'pronouns': pronouns, 'documents': documents_count, 'words': tokens_count
            }
            encoded_content = msgspec.json.encode(content).decode('utf-8')
            destination_file.write(encoded_content)

    @classmethod
    def cli(
        cls,
        num_workers: int = multiprocessing.cpu_count(),
        debug: bool = False,
        **process_single_kwargs: Any
    ) -> None:

        with TemporaryDirectory() as tempdir:
            documents = "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_*/*.gz"
            stats = "s3://ai2-llm/stats/olmo-mix/v1/web/common-crawl"
            metadata = os.path.join(tempdir, "common-crawl")

            processor = cls(
                source_prefix=documents,
                destination_prefix=stats,
                metadata_prefix=metadata,
                num_processes=num_workers,
                debug=debug
            )
            processor(**process_single_kwargs)


class C4InputSpec(InputSpec):
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)


@Registry.add
class c4(BaseStatsProcessor):
    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue[Union[Tuple[int, ...], None]],
        **kwargs: Any
    ):
        decontamination_path = source_path.replace("/documents/", "/attributes/decontamination/")

        documents_decoder = msgspec.json.Decoder(C4InputSpec)
        attributes_decoder = msgspec.json.Decoder(OutputSpec)

        domains: Dict[str, int] = {}
        documents_count = flush_counter = 0
        tokens_count = 0
        pronouns: Dict[str, int] = dict((prs, 0) for pr in PRONOUNS for prs in pr)

        interval = 10_000
        max_dict_size = 100_000
        flush_every = 250_000

        with smart_open.open(source_path, "rb") as source_file, \
                smart_open.open(decontamination_path, "rb") as attributes_file:

            for source_line, attributes_line in zip(source_file, attributes_file):
                document = documents_decoder.decode(source_line)
                attributes = attributes_decoder.decode(attributes_line)

                text = document.text
                for start, end, _ in sorted(
                    attributes.attributes.get('bff_duplicate_paragraph_spans_decontamination', []),
                    key=lambda t: -t[1]
                ):
                    # remove duplicate
                    text = text[:start] + text[end:]

                if not (text := text.strip()):
                    continue

                url = tldextract.extract(document.metadata['url'])
                domain = '.'.join(url[1:])

                documents_count += 1

                if domain in domains:
                    domains[domain] += 1
                elif len(domains) < max_dict_size:
                    domains[domain] = 1
                elif flush_counter >= flush_every:
                    # remove all elements in domains that have <= 1 occurrence
                    domains = {k: v for k, v in domains.items() if v > 1}
                    flush_counter = 0
                else:
                    flush_counter += 1

                for w in blingfire.text_to_words(text).split():
                    tokens_count += 1
                    if (w := w.lower()) in pronouns:
                        pronouns[w] += 1

                if documents_count % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=documents_count % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            content = {
                'domains': domains, 'pronouns': pronouns, 'documents': documents_count, 'words': tokens_count
            }
            encoded_content = msgspec.json.encode(content).decode('utf-8')
            destination_file.write(encoded_content)

    @classmethod
    def cli(
        cls,
        num_workers: int = multiprocessing.cpu_count(),
        debug: bool = False,
        **process_single_kwargs: Any
    ) -> None:
        with TemporaryDirectory() as tempdir:
            documents = "s3://ai2-llm/pretraining-data/sources/c4/v0/documents/train/*.gz"
            stats = "s3://ai2-llm/stats/olmo-mix/v1/web/c4"
            metadata = os.path.join(tempdir, "c4")

            processor = cls(
                source_prefix=documents,
                destination_prefix=stats,
                metadata_prefix=metadata,
                num_processes=num_workers,
                debug=debug
            )
            processor(**process_single_kwargs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stat", choices=[name for name, _ in Registry.all()])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    args = ap.parse_args()

    Registry.get(args.stat).cli(num_workers=args.num_workers, debug=args.debug)
