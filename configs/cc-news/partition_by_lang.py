from argparse import ArgumentParser
from contextlib import ExitStack
import os
from queue import Queue
from tempfile import TemporaryDirectory
from typing import Any, Tuple, Union

import msgspec
import smart_open
from dolma.core.parallel import BaseParallelProcessor
from dolma.core.data_types import InputSpecWithMetadataAndAttributes, OutputSpec


class PartitionByLangProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls,
        queue: "Queue[Union[Tuple[int, ...], None]]",
        /,
        files: int = 0,
        skipped: int = 0,
        written: int = 0,
    ):
        return super().increment_progressbar(queue, files=files, skipped=skipped, written=written)

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue,
        **kwargs: Any,
    ):
        """
        This method is called for each file. It reads the file
        line by line, and writes to the destination file only
        if the document is not empty.
        """

        attribute_prefix = kwargs.get("attribute_prefix", None)
        attribute_name = kwargs.get("attribute_name", None)
        lang_min_score = float(kwargs.get("lang_min_score", -1))

        document_parser = msgspec.json.Decoder(InputSpecWithMetadataAndAttributes)
        attribute_parser = msgspec.json.Decoder(OutputSpec)
        encoder = msgspec.json.Encoder()

        assert attribute_prefix is not None, "Attribute prefix is required"
        assert attribute_name is not None, "Attribute name is required"
        assert 0 <= lang_min_score <= 1, "Language min score must be between 0 and 1"

        dest_dir, dest_file = os.path.split(destination_path)

        written = skipped = 0

        with ExitStack() as stack:
            source_file = stack.enter_context(smart_open.open(source_path, mode="rt", encoding="utf-8"))
            language_attribute_path = source_path.replace("/documents/", f"/attributes/{attribute_name}/")
            language_attribute_file = stack.enter_context(
                smart_open.open(language_attribute_path, mode="rt", encoding="utf-8")
            )
            dst_files = {}

            while True:
                raw_doc = source_file.readline()
                raw_attr = language_attribute_file.readline()

                if not raw_doc or not raw_attr:
                    # end of file
                    break

                attr = attribute_parser.decode(raw_attr)

                all_langs = {
                    k.replace(attribute_prefix, ""): v[0][-1]
                    for k, v in attr.attributes.items()
                    if k.startswith(attribute_prefix)
                }

                if all_langs:
                    top_lang, top_score = max(all_langs.items(), key=lambda x: x[1])
                else:
                    top_lang = "unk"
                    top_score = 0

                if top_score < lang_min_score:
                    top_lang = "unk"
                    skipped += 1

                doc = document_parser.decode(raw_doc)
                doc.attributes = {**(doc.attributes or {}), **attr.attributes}

                if top_lang not in dst_files:
                    dir_path = os.path.join(dest_dir, top_lang)
                    os.makedirs(dir_path, exist_ok=True)
                    dst_files[top_lang] = stack.enter_context(
                        smart_open.open(os.path.join(dir_path, dest_file), mode="wt", encoding="utf-8")
                    )

                dst_files[top_lang].write(encoder.encode(doc).decode('utf-8') + "\n")
                written += 1

                if (written + skipped) > 1000:
                    cls.increment_progressbar(queue, written=written, skipped=skipped)
                    written = skipped = 0

        cls.increment_progressbar(queue, written=written, skipped=skipped, files=1)


def parse_args():
    ag = ArgumentParser()
    ag.add_argument("-s", "--source-prefix", type=str, required=True)
    ag.add_argument("-d", "--destination-prefix", type=str, required=True)
    ag.add_argument("-n", "--num-processes", type=int, default=1)
    ag.add_argument("-u", "--debug", action="store_true")
    ag.add_argument("--temp-dir", type=str, default=None)
    ag.add_argument("--attribute-name", type=str, default="glotlid_doc_v3_1e2")
    ag.add_argument("--attribute-prefix", type=str, default="glotlid_doc_v3_1e2__glotlid_doc_v3_1e2__")
    ag.add_argument("--lang-min-score", type=float, default=0.5)
    return ag.parse_args()


def main():
    args = parse_args()

    with TemporaryDirectory(dir=args.temp_dir) as tmpdir:
        # create the processor
        processor = PartitionByLangProcessor(
            source_prefix=args.source_prefix,
            destination_prefix=args.destination_prefix,
            metadata_prefix=tmpdir,
            num_processes=args.num_processes,
            debug=args.debug,
        )

        # run the processor
        processor(
            attribute_name=args.attribute_name,
            attribute_prefix=args.attribute_prefix,
            lang_min_score=args.lang_min_score,
        )


if __name__ == "__main__":
    main()
