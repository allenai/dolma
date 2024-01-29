import argparse
import bisect
import copy
import gzip
import hashlib
import json
import multiprocessing
import os
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Dict, Generator, List, Tuple, Type, TypeVar, Union

import blingfire
import msgspec
import numpy as np
import smart_open
import tldextract
import tqdm
from dolma.core.data_types import InputSpec, OutputSpec
from dolma.core.parallel import BaseParallelProcessor
from dolma.core.paths import glob_path, make_relative, split_path
from dolma.tokenizer import Tokenizer

T = TypeVar("T", bound=Type["BaseStatsProcessor"])

PRONOUNS = (
    ("she", "her", "her", "hers", "herself"),
    ("he", "him", "his", "his", "himself"),
    ("they", "them", "their", "theirs", "themselves"),
    ("ze", "hir", "hir", "hirs", "hirself"),
    ("ze", "zir", "zir", "zirs", "zirself"),
    ("xey", "xem", "xyr", "xyrs", "xemself"),
    ("ae", "aer", "aer", "aers", "aerself"),
    ("e", "em", "eir", "eirs", "emself"),
    ("ey", "em", "eir", "eirs", "eirself"),
    ("fae", "faer", "faer", "faers", "faerself"),
    ("fey", "fem", "feir", "feirs", "feirself"),
    ("hu", "hum", "hus", "hus", "humself"),
    ("it", "it", "its", "its", "itself"),
    ("jee", "jem", "jeir", "jeirs", "jemself"),
    ("kit", "kit", "kits", "kits", "kitself"),
    ("ne", "nem", "nir", "nirs", "nemself"),
    ("peh", "pehm", "peh's", "peh's", "pehself"),
    ("per", "per", "per", "pers", "perself"),
    ("sie", "hir", "hir", "hirs", "hirself"),
    ("se", "sim", "ser", "sers", "serself"),
    ("shi", "hir", "hir", "hirs", "hirself"),
    ("si", "hyr", "hyr", "hyrs", "hyrself"),
    ("they", "them", "their", "theirs", "themself"),
    ("thon", "thon", "thons", "thons", "thonself"),
    ("ve", "ver", "vis", "vis", "verself"),
    ("ve", "vem", "vir", "virs", "vemself"),
    ("vi", "ver", "ver", "vers", "verself"),
    ("vi", "vim", "vir", "virs", "vimself"),
    ("vi", "vim", "vim", "vims", "vimself"),
    ("xie", "xer", "xer", "xers", "xerself"),
    ("xe", "xem", "xyr", "xyrs", "xemself"),
    ("xey", "xem", "xeir", "xeirs", "xemself"),
    ("yo", "yo", "yos", "yos", "yosself"),
    ("ze", "zem", "zes", "zes", "zirself"),
    ("ze", "mer", "zer", "zers", "zemself"),
    ("zee", "zed", "zeta", "zetas", "zedself"),
    ("zie", "zir", "zir", "zirs", "zirself"),
    ("zie", "zem", "zes", "zes", "zirself"),
    ("zie", "hir", "hir", "hirs", "hirself"),
    ("zme", "zmyr", "zmyr", "zmyrs", "zmyrself"),
)


@dataclass
class Domains:
    pages: Dict[str, int] = field(default_factory=dict)
    words: Dict[str, int] = field(default_factory=dict)
    _size: int = 100_000

    def add(self, domain: str, count_words: int, count_pages: int = 1, no_limit: bool = False) -> bool:
        if domain not in self.pages:
            if self._size < len(self.pages) and not no_limit:
                return False
            self.pages[domain] = 0
            self.words[domain] = 0

        self.pages[domain] += count_pages
        self.words[domain] += count_words
        return True

    def shrink(self, to_size: bool = False) -> bool:
        threshold = 1
        if to_size:
            # find the threshold that will keep the top self._size domains
            prob = max((1 - self._size / len(self.pages)) * 100, 0)
            threshold = max(threshold, round(np.percentile(sorted(self.pages.values()), prob)))

        previous_size = len(self.pages)
        self.pages = {k: v for k, v in self.pages.items() if v > threshold}
        self.words = {k: v for k, v in self.words.items() if k in self.pages}
        current_size = len(self.pages)
        return previous_size < current_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pages": {k: v for k, v in sorted(self.pages.items(), key=lambda e: -e[1])},
            "words": {k: v for k, v in sorted(self.words.items(), key=lambda e: -e[1])},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Domains":
        return cls(pages=d["pages"], words=d["words"])

    def merge(self, other: "Domains", inplace: bool = True, shrink: bool = False) -> "Domains":
        to_return = self if inplace else deepcopy(self)

        for page in other.pages:
            to_return.add(domain=page, count_words=other.words[page], count_pages=other.pages[page], no_limit=True)

        if shrink:
            to_return.shrink(to_size=True)

        return to_return


@dataclass
class Counts:
    documents: int = 0
    tokens: int = 0
    domains: Domains = field(default_factory=Domains)
    pronouns: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in chain.from_iterable(PRONOUNS)})
    _flush: int = 250_000
    _current: int = 0

    def shrink(self) -> bool:
        self._current += 1
        if self._current >= self._flush:
            self._current = 0
            self.domains.shrink()
            return True

        return False

    def add(self, text: str, url: str) -> bool:
        if not (text := text.strip()):
            return False

        words = [w.lower() for w in blingfire.text_to_words(text).split()]
        extracted_url = tldextract.extract(url)
        domain = ".".join(extracted_url[1:]).lower()

        for w in words:
            if w in self.pronouns:
                self.pronouns[w] += 1

        self.documents += 1
        self.tokens += len(words)
        self.domains.add(domain=domain, count_words=len(words))

        self.shrink()
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domains": self.domains.to_dict(),
            "pronouns": {k: v for k, v in sorted(self.pronouns.items(), key=lambda e: -e[1])},
            "documents": self.documents,
            "words": self.tokens,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Counts":
        return cls(
            documents=d["documents"],
            tokens=d["words"],
            domains=Domains.from_dict(d["domains"]),
            pronouns=d["pronouns"],
        )

    def merge(self, other: "Counts", inplace: bool = True, shrink: bool = False) -> "Counts":
        (to_return := self if inplace else deepcopy(self)).documents += other.documents
        to_return.tokens += other.tokens
        to_return.domains.merge(other.domains, inplace=True, shrink=shrink)
        for pronoun, count in other.pronouns.items():
            to_return.pronouns[pronoun] += count
        return to_return


class Registry:
    __registry__: Dict[str, Type["BaseStatsProcessor"]] = {}

    @classmethod
    def add(cls, obj: T) -> T:
        cls.__registry__[obj.__name__] = obj
        return obj

    @classmethod
    def get(cls, name: str) -> Type["BaseStatsProcessor"]:
        return cls.__registry__[name]

    @classmethod
    def all(cls) -> Generator[Tuple[str, Type["BaseStatsProcessor"]], None, None]:
        yield from cls.__registry__.items()


class BaseStatsProcessor(BaseParallelProcessor):
    documents: Union[str, List[str]]
    stats: str
    skip_parallel: bool = False

    @classmethod
    def increment_progressbar(
        cls, queue: "Queue[Union[Tuple[int, ...], None]]", /, files: int = 0, documents: int = 0
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @staticmethod
    # read all paths in using threads
    def _read_json(path: str) -> Counts:
        with smart_open.open(path, "rt") as source_file:
            return msgspec.json.decode(source_file.read())

    @classmethod
    def _merge_dicts(cls, d1, d2):
        d1 = copy.deepcopy(d1)
        for k, v in d2.items():
            if isinstance(v, dict):
                d1[k] = cls._merge_dicts(d1.get(k, {}), v)
            else:
                d1[k] = d1.get(k, 0) + v
        return d1

    @classmethod
    def _run_parallel_processor(cls, stats_root: str, num_workers: int, debug: bool, **process_single_kwargs: Any):
        with TemporaryDirectory() as tempdir:
            h = hashlib.md5()
            for path in cls.documents if isinstance(cls.documents, list) else [cls.documents]:
                h.update(path.encode())

            metadata = os.path.join(tempdir, h.hexdigest())
            processor = cls(
                source_prefix=cls.documents,
                destination_prefix=stats_root,
                metadata_prefix=metadata,
                num_processes=num_workers,
                debug=debug,
            )
            if not cls.skip_parallel:
                processor(**process_single_kwargs)

    @staticmethod
    def _group_by_subset(paths: List[str]) -> Dict[str, List[str]]:
        shared, _ = make_relative(paths)
        shared = shared.rstrip("/") + "/"

        grouped_paths: Dict[str, List[str]] = {}
        for path in sorted(paths):
            _, parts = split_path(path.replace(shared, ""))
            grouped_paths.setdefault("/".join(parts[:-1]), []).append(path)
        return grouped_paths

    @classmethod
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        stats_root = cls.stats.split("*", 1)[0].rstrip("/")

        cls._run_parallel_processor(
            stats_root=stats_root,
            num_workers=num_workers,
            debug=debug,
            **process_single_kwargs,
        )

        paths = list(glob_path(cls.stats))
        grouped_paths = cls._group_by_subset(paths)

        grouped_counts: Dict[str, dict] = defaultdict(dict)

        with tqdm.tqdm(desc=f"Merging {cls.__name__} stats", unit=" files", total=len(paths)) as pbar:
            for subset, sub_paths in grouped_paths.items():
                with multiprocessing.Pool(num_workers) as pool:
                    if debug:
                        data = (cls._read_json(path) for path in sub_paths)
                    else:
                        data = (e for e in pool.imap(cls._read_json, sub_paths))

                    for content in data:
                        pbar.update(1)
                        grouped_counts[subset] = cls._merge_dicts(grouped_counts[subset], content)

        global_counts: dict = {}
        for subset_count in grouped_counts.values():
            for k, v in cls._merge_dicts(global_counts, subset_count).items():
                global_counts[k] = v
        grouped_counts["__GLOBAL__"] = global_counts

        summary_dest = f"{stats_root}/summary.json"
        with smart_open.open(summary_dest, "wt") as destination_file:
            destination_file.write(json.dumps(grouped_counts, indent=2, sort_keys=True))


@Registry.add
class books(BaseStatsProcessor):
    documents = "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/books/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v1/books/gutenberg/*.gz"

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # for the data sheet, what statistics you think we should include? I could
        # do # of docs, # tokens, distribution of URLs, pronouns, s2 FOS, stack
        # languages?
        decoder = msgspec.json.Decoder(InputSpec)
        documents = words = 0
        interval = 10_000

        with smart_open.open(source_path, "rb") as source_file:
            for line in source_file:
                document = decoder.decode(line)
                documents += 1
                words += len(blingfire.text_to_words(document.text).split())

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=documents % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps({"documents": documents, "words": words}, indent=2))


@Registry.add
class wiki(books):
    documents = "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/wiki/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v1/wiki/en_simple/*.gz"


@Registry.add
class cc_v1(BaseStatsProcessor):
    documents = "s3://ai2-llm/pretraining-data/sources/common-crawl/v1/documents/cc_en_*/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v1/cc/v1/**/*.gz"

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attributes = [source_path.replace("/documents/", "/attributes/c4_rules/")]

        # for the data sheet, what statistics you think we should include? I could
        # do # of docs, # tokens, distribution of URLs, pronouns, s2 FOS, stack languages?
        doc_decoder = msgspec.json.Decoder(InputSpec)
        attr_decoder = msgspec.json.Decoder(OutputSpec)
        stats = {
            "length": 0,
            "count": 0,
            "c4_count": 0,
            "c4_length": 0,
            "c4_matches": 0,
        }
        documents = 0
        interval = 10_000

        with ExitStack() as stack:
            doc_file = stack.enter_context(smart_open.open(source_path, "rb"))
            atts_files = [stack.enter_context(smart_open.open(path, "rb")) for path in attributes]

            for doc_line, *attr_lines in zip(doc_file, *atts_files):
                doc = doc_decoder.decode(doc_line)
                stats["length"] += len(doc.text)
                stats["count"] += 1

                attrs = {}
                for line in attr_lines:
                    attrs.update(attr_decoder.decode(line).attributes)

                # C4 stats
                c4_removal = attrs.get("c4_rules__c4_v1__lines_with_no_ending_punctuation", [])
                stats["c4_count"] += len(c4_removal)
                stats["c4_length"] += sum(s[-1] for s in c4_removal)
                stats["c4_matches"] += 1 if c4_removal else 0

                documents += 1

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=documents % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(stats, indent=2))


@Registry.add
class just_cc_dedup(BaseStatsProcessor):
    documents = "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_*/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v1/cc/just_cc_dedup/**/*.gz"

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        dedup = source_path.replace("/documents/", "/attributes/dedupe_paragraphs/")

        attr_decoder = msgspec.json.Decoder(OutputSpec)
        stats = {
            "dedupe_paragraphs_count": 0,
            "dedupe_paragraphs_length": 0,
            "dedupe_paragraphs_matches": 0,
        }
        documents = 0
        interval = 10_000

        with ExitStack() as stack:
            try:
                dedup_file = stack.enter_context(smart_open.open(dedup, "rb"))
            except Exception:
                return

            for ln in dedup_file:
                attrs = attr_decoder.decode(ln).attributes

                # Duplicates stats
                dups = [p for p in attrs.get("bff_duplicate_paragraph_spans", []) if p[1] - p[0] > 0]
                stats["dedupe_paragraphs_count"] += len(dups)
                stats["dedupe_paragraphs_length"] += sum(s[1] - s[0] for s in dups)
                stats["dedupe_paragraphs_matches"] += 1 if dups else 0

                documents += 1

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=documents % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(stats, indent=2))


@Registry.add
class dolma_v15r2_counts(BaseStatsProcessor):
    documents = "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5r2/documents/*/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/dolma-v1_5r2/counts/*/*.gz"
    skip_parallel = True

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # for the data sheet, what statistics you think we should include? I could
        # do # of docs, # tokens, distribution of URLs, pronouns, s2 FOS, stack
        # languages?
        decoder = msgspec.json.Decoder(InputSpec)
        documents = words = 0
        olmo_tokens = llama_tokens = 0
        interval = 10_000

        olmo_tokenizer = Tokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")
        llama_tokenizer = Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

        with smart_open.open(source_path, "rb") as source_file:
            for line in source_file:
                document = decoder.decode(line)
                documents += 1
                words += len(blingfire.text_to_words(document.text).split())
                olmo_tokens += len(olmo_tokenizer.encode(document.text, add_special_tokens=False))
                llama_tokens += len(llama_tokenizer.encode(document.text, add_special_tokens=False))

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=documents % interval)

        counters = {
            "documents": documents,
            "words": words,
            "olmo_tokens": olmo_tokens,
            "llama_tokens": llama_tokens,
        }

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(counters, indent=2, sort_keys=True))


@Registry.add
class dolma_v15r2_olmo(BaseStatsProcessor):
    documents = "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1_5r2/documents/*/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/dolma-v1_5r2/counts_with_bytes/*/*.gz"
    skip_parallel = False

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        decoder = msgspec.json.Decoder(InputSpec)
        documents = words = 0
        olmo_tokens = 0
        utf8_length = 0
        bytes_length = 0
        gzip_bytes_length = 0
        interval = 10_000

        olmo_tokenizer = Tokenizer.from_pretrained("allenai/gpt-neox-olmo-dolma-v1_5")

        with smart_open.open(source_path, "rb") as source_file:
            for line in source_file:
                document = decoder.decode(line)
                documents += 1
                words += len(blingfire.text_to_words(document.text).split())
                olmo_tokens += len(olmo_tokenizer.encode(document.text, add_special_tokens=False))
                bytes_length += len(d := document.text.encode("utf-8"))
                utf8_length += len(d.decode("utf-8"))
                gzip_bytes_length += gzip.compress(document.text.encode("utf-8")).__sizeof__()

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=documents % interval)

        counters = {
            "documents": documents,
            "words": words,
            "olmo_tokens": olmo_tokens,
            "bytes_length": bytes_length,
            "gzip_bytes_length": gzip_bytes_length,
            "utf8_length": utf8_length,
        }

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(counters, indent=2, sort_keys=True))


@Registry.add
class cc_v1_c4_cleaned(BaseStatsProcessor):
    documents = "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_*/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v1/cc/v1_c4_cleaned/**/*.gz"
    decontamination_key: str = "decontamination"
    repetitions_threshold = 100

    @classmethod
    def gopher_rules(cls, attrs: Dict[str, List[Tuple[int, int, float]]]) -> List[Tuple[int, int, float]]:
        matching_spans: List[Tuple[int, int, float]] = []

        for span in attrs.get("gopher_rules__gopher_v1__word_count", []):
            if span[2] < 50 or span[2] > 100000:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__median_word_length", []):
            if span[2] < 3 or span[2] > 10:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__symbol_to_word_ratio", []):
            if span[2] > 0.1:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_words_with_alpha_character", []):
            if span[2] < 0.8:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__required_word_count", []):
            if span[2] < 2:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_lines_starting_with_bullet_point", []):
            if span[2] > 0.9:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_lines_ending_with_ellipsis", []):
            if span[2] > 0.3:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_duplicate_lines", []):
            if span[2] > 0.3:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_duplicate_lines", []):
            if span[2] > 0.3:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_most_common_2gram", []):
            if span[2] > 0.2:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_most_common_3gram", []):
            if span[2] > 0.18:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_most_common_4gram", []):
            if span[2] > 0.16:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_duplicate_5grams", []):
            if span[2] > 0.15:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_duplicate_6grams", []):
            if span[2] > 0.14:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_duplicate_7grams", []):
            if span[2] > 0.13:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_duplicate_8grams", []):
            if span[2] > 0.12:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_duplicate_9grams", []):
            if span[2] > 0.11:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        for span in attrs.get("gopher_rules__gopher_v1__fraction_of_characters_in_duplicate_10grams", []):
            if span[2] > 0.10:
                bisect.insort(matching_spans, (span[0], span[1], 1.0))

        return cls._merge_spans(matching_spans)

    @classmethod
    def _merge_spans(cls, matching_spans: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        # merge spans if overlapping
        merged_spans: List[Tuple[int, int, float]] = []
        current_span = None

        for span in matching_spans:
            if span[1] - span[0] <= 0:
                continue
            elif current_span is None:
                current_span = span
            elif span[0] <= current_span[1]:  # type: ignore
                current_span = (current_span[0], span[1], 1.0)
            else:
                merged_spans.append(current_span)
                current_span = span

        if current_span:
            merged_spans.append(current_span)

        return merged_spans

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attributes = [
            source_path.replace("/documents/", "/attributes/gopher_rules/"),
            source_path.replace("/documents/", f"/attributes/{cls.decontamination_key}/"),
            source_path.replace("/documents/", "/attributes/hatespeech_nsfw_cc_v3/"),
            source_path.replace("/documents/", "/attributes/pii_detection/"),
            source_path.replace("/documents/", "/attributes/dedupe_paragraphs/"),
            source_path.replace("/documents/", "/attributes/dedupe_docs_v2/"),
            source_path.replace("/documents/", "/attributes/tokenizer_repetitions_v2r2/"),
        ]

        doc_decoder = msgspec.json.Decoder(InputSpec)
        attr_decoder = msgspec.json.Decoder(OutputSpec)
        stats = {
            "length": 0,
            "count": 0,
            "gopher_count": 0,
            "gopher_length": 0,
            "gopher_matches": 0,
            "decontamination_count": 0,
            "decontamination_length": 0,
            "decontamination_matches": 0,
            "dedupe_paragraphs_count": 0,
            "dedupe_paragraphs_length": 0,
            "dedupe_paragraphs_matches": 0,
            "dedupe_docs_count": 0,
            "dedupe_docs_length": 0,
            "dedupe_docs_matches": 0,
            "repetitions_count": 0,
            "repetitions_length": 0,
            "repetitions_matches": 0,
            "hatespeech_nsfw_count": 0,
            "hatespeech_nsfw_length": 0,
            "hatespeech_nsfw_matches": 0,
            "pii_count": 0,
            "pii_length": 0,
            "pii_matches_le_5": 0,
            "pii_matches_gt_5": 0,
        }
        documents = 0
        interval = 10_000

        with ExitStack() as stack:
            doc_file = stack.enter_context(smart_open.open(source_path, "rb"))

            try:
                atts_files = [stack.enter_context(smart_open.open(path, "rb")) for path in attributes]
            except Exception:
                return

            for doc_line, *attr_lines in zip(doc_file, *atts_files):
                doc = doc_decoder.decode(doc_line)
                stats["length"] += len(doc.text)
                stats["count"] += 1

                attrs = {}
                for line in attr_lines:
                    attrs.update(attr_decoder.decode(line).attributes)

                # Gopher stats
                gopher_removal = cls.gopher_rules(attrs)
                stats["gopher_count"] += len(gopher_removal)
                stats["gopher_length"] += sum(s[1] - s[0] for s in gopher_removal)
                stats["gopher_matches"] += 1 if gopher_removal else 0

                # Deduplication stats
                decontamination_removal = attrs.get("bff_duplicate_paragraph_spans_decontamination", [])
                stats["decontamination_count"] += len(decontamination_removal)
                stats["decontamination_length"] += sum(s[1] - s[0] for s in decontamination_removal)
                stats["decontamination_matches"] += 1 if decontamination_removal else 0

                # jigsaw stats
                jigsaw_match: List[Tuple[int, int, float]] = []
                nsfw = attrs.get("hatespeech_nsfw_cc_v3__jigsaw_nsfw_sencence_v2____label__nsfw", [])
                for span in nsfw:
                    if span[2] > 0.4:
                        bisect.insort(jigsaw_match, (span[0], span[1], 1.0))

                toxic = attrs.get("hatespeech_nsfw_cc_v3__jigsaw_hatespeech_sentence_v2____label__toxic", [])
                for span in toxic:
                    if span[2] > 0.4:
                        bisect.insort(jigsaw_match, (span[0], span[1], 1.0))

                jigsaw_match = cls._merge_spans(jigsaw_match)

                stats["hatespeech_nsfw_count"] += len(jigsaw_match)
                stats["hatespeech_nsfw_length"] += sum(s[1] - s[0] for s in jigsaw_match)
                stats["hatespeech_nsfw_matches"] += 1 if jigsaw_match else 0

                # PII stats
                pii_removal = (
                    attrs.get("pii_detection__pii_regex_with_counts_fast_v2__EMAIL_ADDRESS", [])
                    + attrs.get("pii_detection__pii_regex_with_counts_fast_v2__PHONE_NUMBER", [])
                    + attrs.get("pii_detection__pii_regex_with_counts_fast_v2__IP_ADDRESS", [])
                )
                stats["pii_count"] += len(pii_removal)
                stats["pii_length"] += sum(s[1] - s[0] for s in pii_removal)
                stats["pii_matches_le_5"] += 1 if 0 < len(pii_removal) <= 5 else 0
                stats["pii_matches_gt_5"] += 1 if len(pii_removal) > 5 else 0

                # Duplicates stats
                dups = [p for p in attrs.get("bff_duplicate_paragraph_spans", []) if p[1] - p[0] > 0]
                stats["dedupe_paragraphs_count"] += len(dups)
                stats["dedupe_paragraphs_length"] += sum(s[1] - s[0] for s in dups)
                stats["dedupe_paragraphs_matches"] += 1 if dups else 0

                docs_dups = [p for p in attrs.get("bff_duplicate_docs", []) if p[1] - p[0] > 0]
                stats["dedupe_docs_count"] += len(docs_dups)
                stats["dedupe_docs_length"] += sum(s[1] - s[0] for s in docs_dups)
                stats["dedupe_docs_matches"] += 1 if docs_dups else 0

                # Repetitions stats
                (_, _, max_reps), *_ = attrs.get(
                    "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_score_repetition", [[0, 0, 0]]
                )
                if max_reps >= cls.repetitions_threshold:
                    reps = [
                        r
                        for r in attrs.get(
                            "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__repetition", []
                        )
                        if r[-1] >= cls.repetitions_threshold
                    ]
                    stats["repetitions_count"] += len(reps)
                    stats["repetitions_length"] += len(doc.text)
                    stats["repetitions_matches"] += 1

                documents += 1

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=documents % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(stats, indent=2))


@Registry.add
class v15_cc_c4_cleaned(cc_v1_c4_cleaned):
    documents = "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_*/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v15/cc/v1_c4_cleaned/**/*.gz"
    decontamination_key: str = "perplexity_suite_v3_option2"


@Registry.add
class v15r2_cc_c4_cleaned_dup(cc_v1_c4_cleaned):
    documents = "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_*/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v15/cc/v15r2_cc_c4_cleaned_dup/**/*.gz"
    decontamination_key: str = "perplexity_suite_v3_option2"

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attributes = [
            source_path.replace("/documents/", "/attributes/tokenizer_repetitions_v2r2/"),
            source_path.replace("/documents/", "/attributes/dedupe_paragraphs/"),
            # source_path.replace("/documents/", "/attributes/dedupe_docs/"),
        ]

        doc_decoder = msgspec.json.Decoder(InputSpec)
        attr_decoder = msgspec.json.Decoder(OutputSpec)

        stats = {
            "doc_length": 0,
            "doc_count": 0,
            "repetitions_count": defaultdict(int),
            "repetitions_length": defaultdict(int),
            "repetitions_period": defaultdict(int),
        }
        interval = 10_000

        with ExitStack() as stack:
            doc_file = stack.enter_context(smart_open.open(source_path, "rb"))

            try:
                atts_files = [stack.enter_context(smart_open.open(path, "rb")) for path in attributes]
            except Exception:
                return

            for doc_line, *attr_lines in zip(doc_file, *atts_files):
                doc = doc_decoder.decode(doc_line)
                stats["doc_length"] += len(doc.text)
                stats["doc_count"] += 1

                attrs = {}
                for line in attr_lines:
                    attrs.update(attr_decoder.decode(line).attributes)

                repetitions = attrs.get("tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__repetition", [])
                stats["repetitions_count"][len(repetitions)] += 1

                repetition_max_length = attrs.get(
                    "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_length_repetition",
                    [[0, 0, 0]],
                )[0][-1]
                stats["repetitions_length"][repetition_max_length] += 1

                repetitions_period = attrs.get(
                    "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_score_repetition",
                    [[0, 0, 0]],
                )[0][-1]
                stats["repetitions_period"][repetitions_period] += 1

                if stats["doc_count"] % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=stats["doc_count"] % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(stats, indent=2))


@Registry.add
class LineStatsCC(cc_v1_c4_cleaned):
    # Selection of documents:
    # import random; print([random.randint(0, 1334) for _ in range(10)])
    documents = [
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-0700.json.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-0724.json.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-0788.json.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-1286.json.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-0600.json.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-0752.json.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-0239.json.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-1270.json.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-0786.json.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/cc_en_head-0857.json.gz",
    ]
    stats = [
        "./cc_en_head-0700-stats.json.gz",
        "./cc_en_head-0724-stats.json.gz",
        "./cc_en_head-0788-stats.json.gz",
        "./cc_en_head-1286-stats.json.gz",
        "./cc_en_head-0600-stats.json.gz",
        "./cc_en_head-0752-stats.json.gz",
        "./cc_en_head-0239-stats.json.gz",
        "./cc_en_head-1270-stats.json.gz",
        "./cc_en_head-0786-stats.json.gz",
        "./cc_en_head-0857-stats.json.gz",
    ]
    decontamination_key: str = "decontamination"

    @classmethod
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        cls._run_parallel_processor(
            stats_root=cls.stats,
            num_workers=num_workers,
            debug=debug,
            **process_single_kwargs,
        )

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attributes = [
            source_path.replace("/documents/", "/attributes/gopher_rules/"),
            source_path.replace("/documents/", f"/attributes/{cls.decontamination_key}/"),
            source_path.replace("/documents/", "/attributes/hatespeech_nsfw_cc_v3/"),
            source_path.replace("/documents/", "/attributes/pii_detection/"),
            source_path.replace("/documents/", "/attributes/dedupe_paragraphs/"),
        ]

        doc_decoder = msgspec.json.Decoder(InputSpec)
        attr_decoder = msgspec.json.Decoder(OutputSpec)
        documents = 0
        interval = 10_000

        with ExitStack() as stack:
            doc_file = stack.enter_context(smart_open.open(source_path, "rb"))
            out_file = stack.enter_context(smart_open.open(destination_path, "wt"))

            try:
                atts_files = [stack.enter_context(smart_open.open(path, "rb")) for path in attributes]
            except Exception:
                return

            for doc_line, *attr_lines in zip(doc_file, *atts_files):
                doc = doc_decoder.decode(doc_line)
                attrs = {}
                for line in attr_lines:
                    attrs.update(attr_decoder.decode(line).attributes)
                out_line = {}

                # Gopher stats
                gopher_removal = cls.gopher_rules(attrs)
                out_line["gopher_spans"] = gopher_removal

                # Deduplication stats
                decontamination_removal = attrs.get("bff_duplicate_paragraph_spans_decontamination", [])
                out_line["decontamination_spans"] = decontamination_removal

                # jigsaw stats
                jigsaw_match: List[Tuple[int, int, float]] = []
                nsfw = attrs.get("hatespeech_nsfw_cc_v3__jigsaw_nsfw_sencence_v2____label__nsfw", [])
                for span in nsfw:
                    if span[2] > 0.4:
                        bisect.insort(jigsaw_match, (span[0], span[1], 1.0))

                toxic = attrs.get("hatespeech_nsfw_cc_v3__jigsaw_hatespeech_sentence_v2____label__toxic", [])
                for span in toxic:
                    if span[2] > 0.4:
                        bisect.insort(jigsaw_match, (span[0], span[1], 1.0))

                jigsaw_match = cls._merge_spans(jigsaw_match)
                out_line["hatespeech_spans"] = jigsaw_match

                # PII stats
                pii_removal = (
                    attrs.get("pii_detection__pii_regex_with_counts_fast_v2__EMAIL_ADDRESS", [])
                    + attrs.get("pii_detection__pii_regex_with_counts_fast_v2__PHONE_NUMBER", [])
                    + attrs.get("pii_detection__pii_regex_with_counts_fast_v2__IP_ADDRESS", [])
                )
                out_line["pii_spans"] = pii_removal

                # Duplicates stats
                dups = [p for p in attrs.get("bff_duplicate_paragraph_spans", []) if p[1] - p[0] > 0]
                out_line["dedupe_paragraphs_spans"] = dups

                documents += 1

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

                out_file.write(json.dumps(out_line) + "\n")

        cls.increment_progressbar(queue, files=1, documents=documents % interval)


class C4InputSpec(InputSpec):
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)


@Registry.add
class c4(BaseStatsProcessor):
    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attrs_path = source_path.replace("/documents/", "/attributes/decontamination/")

        documents_decoder = msgspec.json.Decoder(C4InputSpec)
        attributes_decoder = msgspec.json.Decoder(OutputSpec)
        counts = Counts()
        interval = 10_000

        with smart_open.open(source_path, "rb") as doc_file, smart_open.open(attrs_path, "rb") as attrs_file:
            for source_line, attributes_line in zip(doc_file, attrs_file):
                document = documents_decoder.decode(source_line)
                attributes = attributes_decoder.decode(attributes_line)

                text = document.text
                for start, end, _ in sorted(
                    attributes.attributes.get("bff_duplicate_paragraph_spans_decontamination", []),
                    key=lambda t: -t[1],
                ):
                    # remove duplicate
                    text = text[:start] + text[end:]

                counts.add(text=text, url=document.metadata["url"])

                if counts.documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=counts.documents % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(counts.to_dict(), indent=2))

    @classmethod
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        with TemporaryDirectory() as tempdir:
            documents = "s3://ai2-llm/pretraining-data/sources/c4/v0/documents/train/*.gz"
            stats = "s3://ai2-llm/stats/olmo-mix/v1/web/c4"
            metadata = os.path.join(tempdir, "c4")

            processor = cls(
                source_prefix=documents,
                destination_prefix=stats,
                metadata_prefix=metadata,
                num_processes=num_workers,
                debug=debug,
            )
            processor(**process_single_kwargs)


@Registry.add
class s2(BaseStatsProcessor):
    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attrs_path = source_path.replace("/documents/", "/attributes/decontamination/")

        documents_decoder = msgspec.json.Decoder(C4InputSpec)
        attributes_decoder = msgspec.json.Decoder(OutputSpec)

        interval = 10_000

        counts: dict = {
            f: {"year": {}, "s2fos": {}, "documents": 0, "tokens": 0} for f in ["full_text", "abstract"]
        }
        key = "full_text" if "dataset=s2orc" in source_path else "abstract"
        cnt = 0

        with smart_open.open(source_path, "rb") as doc_file, smart_open.open(attrs_path, "rb") as attrs_file:
            for source_line, attributes_line in zip(doc_file, attrs_file):
                cnt += 1

                document = documents_decoder.decode(source_line)
                attributes = attributes_decoder.decode(attributes_line)

                text = document.text
                for start, end, _ in sorted(
                    attributes.attributes.get("bff_duplicate_paragraph_spans_decontamination", []),
                    key=lambda t: -t[1],
                ):
                    # remove duplicate
                    text = text[:start] + text[end:]

                if not (text := text.strip()):
                    continue

                counts[key]["documents"] += 1
                counts[key]["tokens"] += len(blingfire.text_to_words(text).split())

                if document.metadata["year"] not in counts[key]["year"]:
                    counts[key]["year"][document.metadata["year"]] = 0
                counts[key]["year"][document.metadata["year"]] += 1

                if len(document.metadata["s2fieldsofstudy"]) == 0:
                    document.metadata["s2fieldsofstudy"] = ["null"]

                for fos in document.metadata["s2fieldsofstudy"]:
                    if fos not in counts[key]["s2fos"]:
                        counts[key]["s2fos"][fos] = 0
                    counts[key]["s2fos"][fos] += 1

                if cnt % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=cnt % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(counts, indent=2))

    @classmethod
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        with TemporaryDirectory() as tempdir:
            documents = (
                "s3://ai2-llm/pretraining-data/sources/s2/v3-fos/documents/dataset=*/split=train/part_id=*/*.gz"
            )
            stats = "s3://ai2-llm/stats/olmo-mix/v1/papers/peS2o"
            metadata = os.path.join(tempdir, "s2")

            processor = cls(
                source_prefix=documents,
                destination_prefix=stats,
                metadata_prefix=metadata,
                num_processes=num_workers,
                debug=debug,
            )
            processor(**process_single_kwargs)


@Registry.add
class reddit(BaseStatsProcessor):
    repetitions_threshold = 100
    documents = "s3://ai2-llm/pretraining-data/sources/reddit/v5-dedupe-pii-nsfw-toxic/documents/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v1_5/forums/reddit/grouped/*.gz"

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attrs_path = source_path.replace(
            "/documents/",
            "/attributes/tokenizer_repetitions_v2r2/",
        )

        documents_decoder = msgspec.json.Decoder(C4InputSpec)
        attributes_decoder = msgspec.json.Decoder(OutputSpec)

        interval = 10_000

        stats = {
            "length": 0,
            "count": 0,
            "tokens": 0,
            "repetitions_count": 0,
            "repetitions_length": 0,
            "repetitions_matches": 0,
        }
        cnt = 0

        with smart_open.open(source_path, "rb") as doc_file, smart_open.open(attrs_path, "rb") as attrs_file:
            for source_line, attributes_line in zip(doc_file, attrs_file):
                cnt += 1

                document = documents_decoder.decode(source_line)
                attributes = attributes_decoder.decode(attributes_line)
                text = document.text

                if not (text := text.strip()):
                    continue

                stats["count"] += 1
                stats["tokens"] += len(blingfire.text_to_words(text).split())
                stats["length"] += len(text)

                (_, _, max_reps), *_ = attributes.attributes.get(
                    "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__doc_max_score_repetition", [[0, 0, 0]]
                )
                if max_reps >= cls.repetitions_threshold:
                    reps = [
                        r
                        for r in attributes.attributes.get(
                            "tokenizer_repetitions_v2r2__tokenizer_repetitions_v2r2__repetition", []
                        )
                        if r[-1] >= cls.repetitions_threshold
                    ]
                    stats["repetitions_count"] += len(reps)
                    stats["repetitions_length"] += sum(s[1] - s[0] for s in reps)
                    stats["repetitions_matches"] += 1

                if cnt % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=cnt % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(stats, indent=2))


class StackInputSpec(InputSpec):
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)
    attributes: Dict[str, Any] = msgspec.field(default_factory=dict)


@Registry.add
class stack_v2(BaseStatsProcessor):
    documents = "s3://ai2-llm/pretraining-data/sources/stack-dedup/v1-mixer/documents/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v1/stack/v1-mixer/*.gz"

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attrs_basic = source_path.replace("/documents/", "/attributes/basic/")
        attrs_code_secrets = source_path.replace("/documents/", "/attributes/rpj-heuristics/")
        # attrs_dedupe_documents = source_path.replace("/documents/", "/attributes/dedupe_documents/")
        # attrs_pii = source_path.replace("/documents/", "/attributes/pii/")

        documents_decoder = msgspec.json.Decoder(StackInputSpec)
        attributes_decoder = msgspec.json.Decoder(OutputSpec)

        interval = 10_000

        counts: dict = {
            "extension": defaultdict(int),
            "license": defaultdict(int),
            "length": 0,
            "count": 0,
            "rpj_count": 0,
            "rpj_length": 0,
            "rpj_matches": 0,
        }
        cnt = 0

        with ExitStack() as stack:
            doc_file = stack.enter_context(smart_open.open(source_path, "rb"))
            attributes_files = [
                stack.enter_context(smart_open.open(attrs_basic, "rb")),
                stack.enter_context(smart_open.open(attrs_code_secrets, "rb")),
                # stack.enter_context(smart_open.open(attrs_dedupe_documents, "rb")),
                # stack.enter_context(smart_open.open(attrs_pii, "rb")),
            ]

            # with smart_open.open(source_path, "rb") as doc_file, smart_open.open(attrs_path, "rb") as attrs_file:
            for source_line, *attributes_line in zip(doc_file, *attributes_files):
                cnt += 1

                doc = documents_decoder.decode(source_line)
                for ln in attributes_line:
                    attributes = attributes_decoder.decode(ln)
                    doc.attributes.update(attributes.attributes)

                if doc.attributes["basic__random_number_v1__random"][-1][-1] > 0.996:
                    # test set; see
                    # https://github.com/allenai/LLM/blob/642d0fad3fb2efd816af507250c4c65c8678cb44/pretrain_data/the_stack/v2-mixer/ablations/v2-mixer-held-out.json#L15
                    continue

                spans = []
                if doc.attributes["rpj_heuristics__code_redpajama_taggers_v1__max_line_length_doc"][0][2] > 1000:
                    spans.append((0, len(doc.text), 1.0))
                elif doc.attributes["rpj_heuristics__code_redpajama_taggers_v1__avg_line_length_doc"][0][2] > 100:
                    spans.append((0, len(doc.text), 1.0))
                elif doc.attributes["rpj_heuristics__code_redpajama_taggers_v1__alnum_prop_doc"][0][2] < 0.25:
                    spans.append((0, len(doc.text), 1.0))
                elif doc.attributes["rpj_heuristics__code_redpajama_taggers_v1__alpha_token_prop_doc"][0][2] < 1.5:
                    spans.append((0, len(doc.text), 1.0))

                counts["rpj_count"] += len(spans)
                counts["rpj_length"] += sum(s[1] - s[0] for s in spans)
                counts["rpj_matches"] += 1 if spans else 0

                counts["count"] += 1
                counts["length"] += len(doc.text)

                counts["extension"][doc.metadata["ext"]] += 1
                for license in doc.metadata["max_forks_repo_licenses"]:
                    counts["license"][license] += 1

                cnt += 1

                if cnt % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=cnt % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(counts, indent=2))


@Registry.add
class stack_v3(stack_v2):
    documents = "s3://ai2-llm/pretraining-data/sources/stack-dedup/v2-mixer/documents/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v1/stack/v2-mixer/*.gz"

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attrs_basic = source_path.replace("/documents/", "/attributes/basic/")
        # attrs_code_secrets = source_path.replace("/documents/", "/attributes/rpj-heuristics/")
        attrs_dedupe_documents = source_path.replace("/documents/", "/attributes/dedupe_documents/")
        attrs_pii = source_path.replace("/documents/", "/attributes/pii/")

        documents_decoder = msgspec.json.Decoder(StackInputSpec)
        attributes_decoder = msgspec.json.Decoder(OutputSpec)

        interval = 10_000

        counts: dict = {
            "extension": defaultdict(int),
            "license": defaultdict(int),
            "length": 0,
            "count": 0,
            "rpj_count": 0,
            "rpj_length": 0,
            "rpj_matches": 0,
        }
        cnt = 0

        with ExitStack() as stack:
            doc_file = stack.enter_context(smart_open.open(source_path, "rb"))
            attributes_files = [
                stack.enter_context(smart_open.open(attrs_basic, "rb")),
                stack.enter_context(smart_open.open(attrs_dedupe_documents, "rb")),
                stack.enter_context(smart_open.open(attrs_pii, "rb")),
            ]

            # with smart_open.open(source_path, "rb") as doc_file, smart_open.open(attrs_path, "rb") as attrs_file:
            for source_line, *attributes_line in zip(doc_file, *attributes_files):
                cnt += 1

                doc = documents_decoder.decode(source_line)
                for ln in attributes_line:
                    attributes = attributes_decoder.decode(ln)
                    doc.attributes.update(attributes.attributes)

                if doc.attributes["basic__random_number_v1__random"][-1][-1] > 0.996:
                    # test set; see
                    # https://github.com/allenai/LLM/blob/642d0fad3fb2efd816af507250c4c65c8678cb44/pretrain_data/the_stack/v2-mixer/ablations/v2-mixer-held-out.json#L15
                    continue

                spans = []
                if doc.attributes["rpj_heuristics__code_redpajama_taggers_v1__max_line_length_doc"][0][2] > 1000:
                    spans.append((0, len(doc.text), 1.0))
                elif doc.attributes["rpj_heuristics__code_redpajama_taggers_v1__avg_line_length_doc"][0][2] > 100:
                    spans.append((0, len(doc.text), 1.0))
                elif doc.attributes["rpj_heuristics__code_redpajama_taggers_v1__alnum_prop_doc"][0][2] < 0.25:
                    spans.append((0, len(doc.text), 1.0))
                elif doc.attributes["rpj_heuristics__code_redpajama_taggers_v1__alpha_token_prop_doc"][0][2] < 1.5:
                    spans.append((0, len(doc.text), 1.0))

                counts["rpj_count"] += len(spans)
                counts["rpj_length"] += sum(s[1] - s[0] for s in spans)
                counts["rpj_matches"] += 1 if spans else 0

                counts["count"] += 1
                counts["length"] += len(doc.text)

                counts["extension"][doc.metadata["ext"]] += 1
                for license in doc.metadata["max_forks_repo_licenses"]:
                    counts["license"][license] += 1

                cnt += 1

                if cnt % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=cnt % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(counts, indent=2))


@Registry.add
class stack_v4(stack_v2):
    documents = "s3://ai2-llm/pretraining-data/sources/stack-dedup/v4-train/documents/*/*.gz"
    stats = "s3://ai2-llm/stats/olmo-mix/v1_5/stack/v4-train/**/*.gz"

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        attrs_basic = source_path.replace("/documents/", "/attributes/perplexity_suite_v3_option2/")
        # attrs_code_secrets = source_path.replace("/documents/", "/attributes/rpj-heuristics/")
        # attrs_dedupe_documents = source_path.replace("/documents/", "/attributes/dedupe_documents/")
        # attrs_pii = source_path.replace("/documents/", "/attributes/pii/")

        documents_decoder = msgspec.json.Decoder(StackInputSpec)
        attributes_decoder = msgspec.json.Decoder(OutputSpec)

        interval = 10_000

        counts: dict = {
            "extension": defaultdict(int),
            "license": defaultdict(int),
            "length": 0,
            "count": 0,
            "decontamination_count": 0,
            "decontamination_length": 0,
            "decontamination_matches": 0,
        }
        cnt = 0

        with ExitStack() as stack:
            doc_file = stack.enter_context(smart_open.open(source_path, "rb"))
            attributes_files = [
                stack.enter_context(smart_open.open(attrs_basic, "rb")),
            ]

            # with smart_open.open(source_path, "rb") as doc_file, smart_open.open(attrs_path, "rb") as attrs_file:
            for source_line, *attributes_line in zip(doc_file, *attributes_files):
                cnt += 1

                doc = documents_decoder.decode(source_line)
                for ln in attributes_line:
                    attributes = attributes_decoder.decode(ln)
                    doc.attributes.update(attributes.attributes)

                attrs = {}
                for line in attributes_line:
                    attrs.update(attributes_decoder.decode(line).attributes)

                # Deduplication stats
                decontamination_removal = attrs.get("bff_duplicate_paragraph_spans_decontamination", [])
                counts["decontamination_count"] += len(decontamination_removal)
                counts["decontamination_length"] += sum(s[1] - s[0] for s in decontamination_removal)
                counts["decontamination_matches"] += 1 if decontamination_removal else 0

                counts["count"] += 1
                counts["length"] += len(doc.text)

                counts["extension"][doc.metadata["ext"]] += 1
                for license in doc.metadata["max_forks_repo_licenses"]:
                    counts["license"][license] += 1

                cnt += 1

                if cnt % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=cnt % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(counts, indent=2))


@Registry.add
class LineStatsStack(BaseStatsProcessor):
    # Testing
    # documents = "s3://ai2-llm/pretraining-data/sources/stack-dedup/v0/documents/abap/data_0000.jsonl.gz"
    # stats = "/data/niklas/dolma/abap/data_0000.json.gz"
    documents = "s3://ai2-llm/pretraining-data/sources/stack-dedup/v0/documents/*/*.gz"
    stats = "/data/niklas/dolma/stack"

    @classmethod
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        stats_root = cls.stats

        cls._run_parallel_processor(
            stats_root=stats_root,
            num_workers=num_workers,
            debug=debug,
            **process_single_kwargs,
        )

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: "Queue[Union[Tuple[int, ...], None]]", **kwargs: Any
    ):
        # E.g. `s3://ai2-llm/pretraining-data/sources/stack-dedup/v0/attributes/paper_analysis/abap/`
        attr_path = source_path.replace("/documents/", "/attributes/paper_analysis/")

        doc_decoder = msgspec.json.Decoder(InputSpec)
        attr_decoder = msgspec.json.Decoder(OutputSpec)
        documents = 0
        interval = 10_000

        with ExitStack() as stack:
            doc_file = stack.enter_context(smart_open.open(source_path, "rb"))
            out_file = stack.enter_context(smart_open.open(destination_path, "wt"))

            try:
                attr_file = stack.enter_context(smart_open.open(attr_path, "rb"))
            except Exception as e:
                print(e)
                return

            for doc_line, attrs in zip(doc_file, attr_file):
                doc = doc_decoder.decode(doc_line)
                attrs = attr_decoder.decode(attrs).attributes
                out_line = {}

                ## RPJ ##
                if (
                    (attrs["paper_analysis__code_redpajama_taggers_v1__max_line_length_doc"][0][2] > 1000)
                    or (attrs["paper_analysis__code_redpajama_taggers_v1__avg_line_length_doc"][0][2] > 100)
                    or (attrs["paper_analysis__code_redpajama_taggers_v1__alnum_prop_doc"][0][2] < 0.25)
                    or (attrs["paper_analysis__code_redpajama_taggers_v1__alpha_token_prop_doc"][0][2] < 1.5)
                ):
                    out_line["rpj"] = 1
                else:
                    out_line["rpj"] = 0

                ## StarCoder ##
                if (
                    (attrs["paper_analysis__code_starcoder_taggers_v2__has_xml_template_doc"][0][2] > 0.0)
                    or (attrs["paper_analysis__code_starcoder_taggers_v2__code_to_comment_ratio_doc"][0][2] > 0.8)
                    or (
                        attrs["paper_analysis__code_starcoder_taggers_v2__code_to_comment_ratio_doc"][0][2] <= 0.01
                    )
                    or (
                        any(x in source_path for x in ["python", "java", "javascript"])
                        and (
                            attrs["paper_analysis__code_starcoder_taggers_v2__code_to_text_ratio_html_doc"][0][2]
                            <= 0.1
                        )
                    )
                ):
                    out_line["starcoder"] = 1
                else:
                    out_line["starcoder"] = 0

                documents += 1

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

                out_file.write(json.dumps(out_line) + "\n")

        cls.increment_progressbar(queue, files=1, documents=documents % interval)


@Registry.add
class decon_ppl_v3(BaseStatsProcessor):
    documents = [
        "s3://ai2-llm/pretraining-data/sources/gutenberg/v0/documents/*.gz",
        "s3://ai2-llm/pretraining-data/sources/c4/v0/documents/train/*.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_head/*.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_middle/*.gz",
        "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_tail/*.gz",
        "s3://ai2-llm/pretraining-data/sources/s2/v3/documents/dataset=s2ag/split=train/*/*.gz",
        "s3://ai2-llm/pretraining-data/sources/s2/v3/documents/dataset=s2orc/split=train/*/*.gz",
        "s3://ai2-llm/pretraining-data/sources/reddit/v5-dedupe-pii-nsfw-toxic/documents/*.gz",
        "s3://ai2-llm/pretraining-data/sources/stack-dedup/v4-train/documents/*/*.gz",
        "s3://ai2-llm/pretraining-data/sources/wikipedia/v0/documents/lang=en/*.gz",
        "s3://ai2-llm/pretraining-data/sources/wikipedia/v0/documents/lang=simple/*.gz",
        "s3://ai2-llm/pretraining-data/sources/wikibooks/v0/documents/lang=en/*.gz",
        "s3://ai2-llm/pretraining-data/sources/wikibooks/v0/documents/lang=simple/*.gz",
    ]
    stats = "s3://ai2-llm/stats/olmo-mix/v1_5/decontamination_ppl_v3_option2"


# # # BELOW HERE: AGGREGATION # # #


@Registry.add
class web(BaseStatsProcessor):
    @staticmethod
    # read all paths in using threads
    def _read_json(path: str) -> Counts:
        with smart_open.open(path, "rt") as source_file:
            content = msgspec.json.decode(source_file.read())
            return Counts.from_dict(content)

    @classmethod
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        paths = list(
            chain(
                glob_path("s3://ai2-llm/stats/olmo-mix/v1/web/c4/*"),
                glob_path("s3://ai2-llm/stats/olmo-mix/v1/web/common-crawl/**/*"),
            )
        )
        assert len(paths), "Run c4 and common-crawl first"

        with multiprocessing.Pool(num_workers) as pool:
            data = (cls._read_json(path) for path in paths) if debug else pool.imap(cls._read_json, paths)
            counts = Counts()

            for content in tqdm.tqdm(data, desc="Merging web stats", unit=" files", total=len(paths)):
                counts.merge(content, shrink=True)

        with smart_open.open("s3://ai2-llm/stats/olmo-mix/v1/web/summary.json", "wt") as destination_file:
            destination_file.write(json.dumps(counts.to_dict(), indent=2))


@Registry.add
class papers(BaseStatsProcessor):
    @staticmethod
    # read all paths in using threads
    def _read_json(path: str) -> dict:
        with smart_open.open(path, "rt") as source_file:
            return json.loads(source_file.read())

    @classmethod
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        paths = list(glob_path("s3://ai2-llm/stats/olmo-mix/v1/papers/peS2o/**/*.gz"))
        assert len(paths), "Run s2 first"

        with multiprocessing.Pool(num_workers) as pool:
            data = (cls._read_json(path) for path in paths) if debug else pool.imap(cls._read_json, paths)
            counts: dict = {
                f: {"year": {}, "s2fos": {}, "documents": 0, "tokens": 0} for f in ["full_text", "abstract"]
            }
            for content in tqdm.tqdm(data, desc="Merging web stats", unit=" files", total=len(paths)):
                for key, values in content.items():
                    counts[key]["documents"] += values["documents"]
                    counts[key]["tokens"] += values["tokens"]

                    for year, count in values["year"].items():
                        if year not in counts[key]["year"]:
                            counts[key]["year"][year] = 0
                        counts[key]["year"][year] += count

                    for fos, count in values["s2fos"].items():
                        if fos not in counts[key]["s2fos"]:
                            counts[key]["s2fos"][fos] = 0
                        counts[key]["s2fos"][fos] += count

        with smart_open.open("s3://ai2-llm/stats/olmo-mix/v1/papers/summary.json", "wt") as destination_file:
            destination_file.write(json.dumps(counts, indent=2))


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

    ap = argparse.ArgumentParser()
    ap.add_argument("stat", choices=[name for name, _ in Registry.all()])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    args = ap.parse_args()

    Registry.get(args.stat).cli(num_workers=args.num_workers, debug=args.debug)
