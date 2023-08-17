import argparse
import bisect
import copy
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
from dolma.core.paths import glob_path

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
        th = 1
        if to_size:
            # find the threshold that will keep the top self._size domains
            p = max((1 - self._size / len(self.pages)) * 100, 0)
            th = max(th, round(np.percentile(sorted(self.pages.values()), p)))

        previous_size = len(self.pages)
        self.pages = {k: v for k, v in self.pages.items() if v > th}
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
        self = self if inplace else deepcopy(self)

        for page in other.pages:
            self.add(domain=page, count_words=other.words[page], count_pages=other.pages[page], no_limit=True)

        if shrink:
            self.shrink(to_size=True)

        return self


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
        self = self if inplace else deepcopy(self)
        self.documents += other.documents
        self.tokens += other.tokens
        self.domains.merge(other.domains, inplace=True, shrink=shrink)
        for pronoun, count in other.pronouns.items():
            self.pronouns[pronoun] += count
        return self


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
    @classmethod
    def increment_progressbar(  # type: ignore[override]
        cls,
        queue: Queue[Union[Tuple[int, ...], None]],
        /,
        files: int = 0,
        documents: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        raise NotImplementedError()


@Registry.add
class common_crawl(BaseStatsProcessor):
    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: Queue[Union[Tuple[int, ...], None]], **kwargs: Any
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # for the data sheet, what statistics you think we should include? I could
        # do # of docs, # tokens, distribution of URLs, pronouns, s2 FOS, stack
        # languages?
        decoder = msgspec.json.Decoder(InputSpec)
        counts = Counts()
        interval = 10_000

        with smart_open.open(source_path, "rb") as source_file:
            for line in source_file:
                document = decoder.decode(line)
                counts.add(text=document.text, url=document.id)

                if counts.documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=counts.documents % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(counts.to_dict(), indent=2))

    @classmethod
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        with TemporaryDirectory() as tempdir:
            documents = "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/common-crawl/cc_en_*/*.gz"
            stats = "s3://ai2-llm/stats/olmo-mix/v1/web/common-crawl"
            metadata = os.path.join(tempdir, "common-crawl")

            processor = cls(
                source_prefix=documents,
                destination_prefix=stats,
                metadata_prefix=metadata,
                num_processes=num_workers,
                debug=debug,
            )
            processor(**process_single_kwargs)


@Registry.add
class books(BaseStatsProcessor):
    documents: str = "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/books/*.gz"
    stats: str = "s3://ai2-llm/stats/olmo-mix/v1/books/gutenberg/*.gz"

    @staticmethod
    # read all paths in using threads
    def _read_json(path: str) -> Counts:
        with smart_open.open(path, "rt") as source_file:
            return msgspec.json.decode(source_file.read())

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: Queue[Union[Tuple[int, ...], None]], **kwargs: Any
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
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        stats_root = cls.stats.split("*", 1)[0].rstrip("/")

        with TemporaryDirectory() as tempdir:
            metadata = os.path.join(tempdir, hashlib.md5(cls.documents.encode()).hexdigest())

            processor = cls(
                source_prefix=cls.documents,
                destination_prefix=stats_root,
                metadata_prefix=metadata,
                num_processes=num_workers,
                debug=debug,
            )
            processor(**process_single_kwargs)

        paths = list(glob_path(cls.stats))
        counts: dict = {}

        with multiprocessing.Pool(num_workers) as pool:
            data = (cls._read_json(path) for path in paths) if debug else pool.imap(cls._read_json, paths)

            for content in tqdm.tqdm(data, desc=f"Merging {cls.__name__} stats", unit=" files", total=len(paths)):
                counts = cls._merge_dicts(counts, content)

        summary_dest = f"{stats_root}/summary.json"
        with smart_open.open(summary_dest, "wt") as destination_file:
            destination_file.write(json.dumps(counts, indent=2, sort_keys=True))


@Registry.add
class wiki(books):
    documents: str = "s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/wiki/*.gz"
    stats: str = "s3://ai2-llm/stats/olmo-mix/v1/wiki/en_simple/*.gz"


@Registry.add
class cc_v1(books):
    documents: str = "s3://ai2-llm/pretraining-data/sources/common-crawl/v1/documents/cc_en_*/*.gz"
    stats: str = "s3://ai2-llm/stats/olmo-mix/v1/cc/v1/**/*.gz"

    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: Queue[Union[Tuple[int, ...], None]], **kwargs: Any
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
class cc_v1_c4_cleaned(books):
    documents: str = "s3://ai2-llm/pretraining-data/sources/common-crawl/v1-c4-cleaned/documents/cc_en_*/*.gz"
    stats: str = "s3://ai2-llm/stats/olmo-mix/v1/cc/v1_c4_cleaned/**/*.gz"

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
        cls, source_path: str, destination_path: str, queue: Queue[Union[Tuple[int, ...], None]], **kwargs: Any
    ):
        attributes = [
            source_path.replace("/documents/", "/attributes/gopher_rules/"),
            source_path.replace("/documents/", "/attributes/decontamination/"),
            source_path.replace("/documents/", "/attributes/hatespeech_nsfw_cc_v3/"),
            source_path.replace("/documents/", "/attributes/pii_detection/"),
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
            "hatespeech_nsfw_count": 0,
            "hatespeech_nsfw_length": 0,
            "hatespeech_nsfw_matches": 0,
            "pii_count": 0,
            "pii_length": 0,
            "pii_matches": 0,
            "pii_matches_gt_5": 0,
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

                # # C4 stats
                # c4_removal = attrs.get('c4_rules__c4_v1__lines_with_no_ending_punctuation', [])
                # stats["c4_count"] += len(c4_removal)
                # stats["c4_length"] += sum(s[-1] for s in c4_removal)
                # stats["c4_matches"] += 1 if c4_removal else 0

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
                stats["pii_matches"] += 1 if pii_removal else 0
                stats["pii_matches_gt_5"] += 1 if len(pii_removal) > 5 else 0

                documents += 1

                if documents % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=documents % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(stats, indent=2))


class C4InputSpec(InputSpec):
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)


@Registry.add
class c4(BaseStatsProcessor):
    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: Queue[Union[Tuple[int, ...], None]], **kwargs: Any
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
        cls, source_path: str, destination_path: str, queue: Queue[Union[Tuple[int, ...], None]], **kwargs: Any
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


class StackInputSpec(InputSpec):
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)
    attributes: Dict[str, Any] = msgspec.field(default_factory=dict)


@Registry.add
class stack(BaseStatsProcessor):
    @classmethod
    def process_single(
        cls, source_path: str, destination_path: str, queue: Queue[Union[Tuple[int, ...], None]], **kwargs: Any
    ):
        attrs_basic = source_path.replace("/documents/", "/attributes/basic/")
        attrs_code_secrets = source_path.replace("/documents/", "/attributes/code_secrets/")
        # attrs_dedupe_documents = source_path.replace("/documents/", "/attributes/dedupe_documents/")
        attrs_pii = source_path.replace("/documents/", "/attributes/pii/")

        documents_decoder = msgspec.json.Decoder(StackInputSpec)
        attributes_decoder = msgspec.json.Decoder(OutputSpec)

        interval = 10_000

        counts: dict = {"extension": defaultdict(int), "license": defaultdict(int), "documents": 0, "tokens": 0}
        cnt = 0

        with ExitStack() as stack:
            doc_file = stack.enter_context(smart_open.open(source_path, "rb"))
            attrs_basic_file = stack.enter_context(smart_open.open(attrs_basic, "rb"))
            attrs_code_secrets_file = stack.enter_context(smart_open.open(attrs_code_secrets, "rb"))
            # attrs_dedupe_documents_file = stack.enter_context(smart_open.open(attrs_dedupe_documents, "rb"))
            attrs_pii_file = stack.enter_context(smart_open.open(attrs_pii, "rb"))

            # with smart_open.open(source_path, "rb") as doc_file, smart_open.open(attrs_path, "rb") as attrs_file:
            for source_line, *attributes_line in zip(
                doc_file,
                attrs_basic_file,
                attrs_code_secrets_file,
                # attrs_dedupe_documents_file,
                attrs_pii_file,
            ):
                cnt += 1

                document = documents_decoder.decode(source_line)
                for ln in attributes_line:
                    attributes = attributes_decoder.decode(ln)
                    document.attributes.update(attributes.attributes)

                if document.attributes["basic__random_number_v1__random"][-1][-1] > 0.996:
                    # test set; see
                    # https://github.com/allenai/LLM/blob/642d0fad3fb2efd816af507250c4c65c8678cb44/pretrain_data/the_stack/v2-mixer/ablations/v2-mixer-held-out.json#L15
                    continue

                counts["documents"] += 1
                counts["tokens"] += len(blingfire.text_to_words(document.text).split())

                counts["extension"][document.metadata["ext"]] += 1
                for license in document.metadata["max_forks_repo_licenses"]:
                    counts["license"][license] += 1

                cnt += 1

                if cnt % interval == 0:
                    cls.increment_progressbar(queue, documents=interval)

        cls.increment_progressbar(queue, files=1, documents=cnt % interval)

        with smart_open.open(destination_path, "wt") as destination_file:
            destination_file.write(json.dumps(counts, indent=2))

    @staticmethod
    # read all paths in using threads
    def _read_json(path: str) -> dict:
        with smart_open.open(path, "rt") as source_file:
            content = msgspec.json.decode(source_file.read())
            return content

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
    def cli(cls, num_workers: int = 1, debug: bool = False, **process_single_kwargs: Any) -> None:
        with TemporaryDirectory() as tempdir:
            documents = "s3://ai2-llm/pretraining-data/sources/stack-dedup/v2-mixer/documents/*.gz"
            stats = "s3://ai2-llm/stats/olmo-mix/v1/code/stack"
            metadata = os.path.join(tempdir, "stack")

            processor = cls(
                source_prefix=documents,
                destination_prefix=stats,
                metadata_prefix=metadata,
                num_processes=num_workers,
                debug=debug,
            )
            processor(**process_single_kwargs)

        paths = list(glob_path(f"{stats}/*.gz"))
        counts: dict = {}

        with multiprocessing.Pool(num_workers) as pool:
            data = (cls._read_json(path) for path in paths) if debug else pool.imap(cls._read_json, paths)

            for content in tqdm.tqdm(data, desc="Merging code stats", unit=" files", total=len(paths)):
                counts = cls._merge_dicts(counts, content)

        with smart_open.open("s3://ai2-llm/stats/olmo-mix/v1/code/summary.json", "wt") as destination_file:
            destination_file.write(json.dumps(counts, indent=2, sort_keys=True))


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
