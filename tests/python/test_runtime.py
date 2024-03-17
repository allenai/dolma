import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
from unittest import TestCase

import smart_open

from dolma.core.runtime import (
    _make_paths_from_prefix,
    _make_paths_from_substitution,
    create_and_run_tagger,
)

LOCAL_DATA = Path(__file__).parent.parent / "data"


class TestRuntimeUtilities(TestCase):
    def test_make_paths_from_substitution(self):
        paths = [
            "s3://bucket/common-crawl/documents/cc_*/*.json.gz",
            "/local/path/to/documents/train/*",
        ]
        new_paths = _make_paths_from_substitution(
            paths=paths,
            find="documents",
            replace="attributes",
        )
        self.assertEqual(new_paths, ["s3://bucket/common-crawl/attributes", "/local/path/to/attributes/train"])

    def test_make_paths_from_prefix(self):
        paths = [
            "s3://bucket/common-crawl/documents/cc_head/*.json.gz",
            "s3://bucket/common-crawl/documents/cc_middle/*.json.gz",
            "s3://bucket/common-crawl/documents/cc_tail/*.json.gz",
        ]
        new_paths = _make_paths_from_prefix(
            paths=paths,
            prefix="s3://bucket/common-crawl/attributes/",
        )
        self.assertEqual(
            new_paths,
            [
                "s3://bucket/common-crawl/attributes/cc_head",
                "s3://bucket/common-crawl/attributes/cc_middle",
                "s3://bucket/common-crawl/attributes/cc_tail",
            ],
        )

        paths = [
            "s3://bucket/common-crawl/documents/*.json.gz",
            "s3://bucket2/c4/documents/**/data/*.json.gz",
        ]
        new_paths = _make_paths_from_prefix(
            paths=paths,
            prefix="/local/path/",
        )
        self.assertEqual(
            new_paths,
            [
                "/local/path/bucket/common-crawl/documents",
                "/local/path/bucket2/c4/documents",
            ],
        )

    def test_runtime_e2e(self):
        documents_path = f"{LOCAL_DATA}/provided/documents/000.json.gz"
        experiment_name = "test"
        taggers = ["c4_v1"]

        with TemporaryDirectory() as temp_dir:
            create_and_run_tagger(
                documents=[documents_path],
                destination=temp_dir,
                taggers=taggers,
                experiment=experiment_name,
                debug=True,
            )
            destination_file = os.path.join(temp_dir, experiment_name, "000.json.gz")
            self.assertTrue(os.path.exists(destination_file))

            with smart_open.open(documents_path, "rt") as f:
                document = [json.loads(ln) for ln in f]

            with smart_open.open(destination_file, "rt") as f:
                attributes = [json.loads(ln) for ln in f]

        self.assertEqual(len(document), len(attributes))

        for d, a in zip(document, attributes):
            self.assertEqual(d["id"], a["id"])
            self.assertTrue(sorted(a.keys()), ["attributes", "id", "source"])

            for key, value in a["attributes"].items():
                parts = key.split("__")
                self.assertEqual(len(parts), 3)
                self.assertEqual(parts[0], experiment_name)
                self.assertTrue(parts[1] in taggers)

                for elem in value:
                    self.assertEqual(len(elem), 3)
                    self.assertTrue(isinstance(elem[0], int))
                    self.assertTrue(isinstance(elem[1], int))
                    self.assertTrue(isinstance(elem[2], float))

                if len(value) == 1:
                    self.assertEqual(value[0][0], 0)
                    self.assertEqual(value[0][1], len(d["text"]))

    def test_alt_src(self):
        taggers = ["c4_v1"]
        experiment_name = "test"

        with TemporaryDirectory() as temp_dir:
            create_and_run_tagger(
                documents=[f"{LOCAL_DATA}/provided/documents/000.json.gz"],
                destination=temp_dir,
                taggers=taggers,
                experiment=experiment_name,
                debug=True,
            )
            destination_file = os.path.join(temp_dir, experiment_name, "000.json.gz")
            self.assertTrue(os.path.exists(destination_file))

            with smart_open.open(destination_file, "rt") as f:
                attributes_full_name = [json.loads(ln) for ln in f]

        with TemporaryDirectory() as temp_dir:
            create_and_run_tagger(
                documents=[f"{LOCAL_DATA}/provided/documents/*"],
                destination=temp_dir,
                taggers=taggers,
                experiment=experiment_name,
                debug=True,
            )
            destination_file = os.path.join(temp_dir, experiment_name, "000.json.gz")
            self.assertTrue(os.path.exists(destination_file))

            with smart_open.open(destination_file, "rt") as f:
                attributes_star_in_path = [json.loads(ln) for ln in f]

        with TemporaryDirectory() as temp_dir:
            create_and_run_tagger(
                documents=[f"{LOCAL_DATA}/provided/documents/"],
                destination=temp_dir,
                taggers=taggers,
                experiment=experiment_name,
                debug=True,
            )
            destination_file = os.path.join(temp_dir, experiment_name, "000.json.gz")
            self.assertTrue(os.path.exists(destination_file))

            with smart_open.open(destination_file, "rt") as f:
                attributes_only_dir = [json.loads(ln) for ln in f]

        self.assertEqual(attributes_full_name, attributes_star_in_path)
        self.assertEqual(attributes_full_name, attributes_only_dir)

    def test_multiple_taggers(self, experiment_name: Optional[str] = None):
        documents_dir = Path(f"{LOCAL_DATA}/provided/documents")
        taggers = ["c4_v1", "gopher_v1"]

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "documents").mkdir(exist_ok=True)

            for path in documents_dir.iterdir():
                # ignore non-json files, like .DS_Store
                if path.suffix.endswith(".gz"):
                    shutil.copy(path, temp_path / "documents" / path.name)

            create_and_run_tagger(
                documents=[f"{temp_dir}/documents/*"],
                taggers=taggers,
                experiment=experiment_name,
                debug=True,
            )

            if experiment_name is None:
                all_attribute_dirs = [temp_path / "attributes" / t for t in taggers]
            else:
                all_attribute_dirs = [temp_path / "attributes" / experiment_name]

            for d in all_attribute_dirs:
                # check if a folder for each tagger was created
                self.assertTrue(os.path.exists(d))

            # collect all attributes for all documents here
            attributes = []

            for fn in (temp_path / "documents").iterdir():
                # collect all attributes for the current document here
                current_attrs: List[dict] = []

                for attr_path in all_attribute_dirs:
                    # check if attribute to corresponding document was created
                    attr_fp = attr_path / fn.name
                    self.assertTrue(attr_fp.exists())

                    if len(current_attrs) == 0:
                        with smart_open.open(attr_fp, "rt") as f:
                            # no attributes for this file name loaded in yet
                            current_attrs = [json.loads(ln) for ln in f]
                    else:
                        with smart_open.open(attr_fp, "rt") as f:
                            for i, attr_doc in enumerate(json.loads(ln) for ln in f):
                                # check if attributes are aligned
                                self.assertTrue(attr_doc["id"] == current_attrs[i]["id"])
                                current_attrs[i]["attributes"].update(attr_doc["attributes"])

                attributes.extend(current_attrs)

            for row in attributes:
                # check if name of attribute files is correct
                attribute_files_names = set(k.split("__")[0] for k in row["attributes"].keys())

                if experiment_name is None:
                    self.assertEqual(attribute_files_names, set(taggers))
                else:
                    self.assertEqual(attribute_files_names, {experiment_name})

                # check if name of taggers is correct
                tagger_names = set(k.split("__")[1] for k in row["attributes"].keys())
                self.assertEqual(tagger_names, set(taggers))

    def test_multiple_with_exp_name(self):
        # same as test_multiple_taggers, but provide an experiment name
        # this is to test failure reported here: https://github.com/allenai/dolma/pull/113
        self.test_multiple_taggers(experiment_name="experiment_name")

    def test_alt_exp(self):
        documents_path = f"{LOCAL_DATA}/provided/documents/000.json.gz"
        taggers = ["c4_v1"]

        with TemporaryDirectory() as temp_dir:
            create_and_run_tagger(
                documents=[documents_path],
                destination=temp_dir,
                taggers=taggers,
                experiment="test",
                debug=True,
            )
            destination_file = os.path.join(temp_dir, "test", "000.json.gz")
            self.assertTrue(os.path.exists(destination_file))

            with smart_open.open(destination_file, "rt") as f:
                attributes_with_exp = [json.loads(ln) for ln in f]

        with TemporaryDirectory() as temp_dir:
            create_and_run_tagger(
                documents=[documents_path],
                destination=temp_dir,
                taggers=taggers,
                experiment=None,
                debug=True,
            )
            destination_file = os.path.join(temp_dir, "c4_v1", "000.json.gz")
            self.assertTrue(os.path.exists(destination_file))

            with smart_open.open(destination_file, "rt") as f:
                attributes_without_exp = [json.loads(ln) for ln in f]

        for row_with_exp, row_without_exp in zip(attributes_with_exp, attributes_without_exp):
            for key_with_exp, key_without_exp in zip(row_with_exp["attributes"], row_without_exp["attributes"]):
                parts_with_exp = key_with_exp.split("__")
                parts_without_exp = key_without_exp.split("__")
                self.assertNotEqual(parts_with_exp[0], parts_without_exp[0])
                self.assertEqual(parts_with_exp[1], parts_without_exp[1])
                self.assertEqual(parts_with_exp[2], parts_without_exp[2])
                self.assertEqual(parts_with_exp[0], "test")
                self.assertEqual(parts_without_exp[0], "c4_v1")

    def test_multiple_files(self):
        documents = ["cc_en_head-0091.jsonl.gz", "cc_en_head-0174.jsonl.gz"]
        taggers = ["char_length_v1"]

        with TemporaryDirectory() as temp_dir:
            documents_base_path = os.path.join(temp_dir, "documents")
            shutil.copytree(f"{LOCAL_DATA}/multiple_files", documents_base_path)
            create_and_run_tagger(
                documents=[f"{documents_base_path}/{f}" for f in documents],
                taggers=taggers,
                debug=True,
            )

            # check that the attributes directory was created
            self.assertTrue(os.path.exists(f"{temp_dir}/attributes"))

            # check that no other directories beside what we need for attributes are created
            self.assertEqual(len(os.listdir(f"{temp_dir}/attributes")), len(taggers))

            for tagger in taggers:
                # check that the tagger directory was created
                self.assertTrue(os.path.exists(f"{temp_dir}/attributes/{tagger}"))
                self.assertTrue(os.path.isdir(f"{temp_dir}/attributes/{tagger}"))

                # check that the number of files in the tagger directory is the same as the number of documents
                self.assertEqual(len(os.listdir(f"{temp_dir}/attributes/{tagger}")), len(documents))

                for document in documents:
                    # check that each document has a corresponding file in the tagger directory
                    self.assertTrue(os.path.exists(f"{temp_dir}/attributes/{tagger}/{document}"))
                    self.assertTrue(os.path.isfile(f"{temp_dir}/attributes/{tagger}/{document}"))

                    # open files, check if they are valid json
                    with smart_open.open(f"{temp_dir}/attributes/{tagger}/{document}", "rt") as f:
                        attributes = [json.loads(ln) for ln in f]

                    # open corresponding documents
                    with smart_open.open(f"{documents_base_path}/{document}", "rt") as f:
                        documents = [json.loads(ln) for ln in f]

                    # check if the number of documents and attributes is the same
                    self.assertEqual(len(documents), len(attributes))

                    for attr, doc in zip(attributes, documents):
                        # check if the id of the document and the attribute is the same
                        self.assertEqual(attr["id"], doc["id"])
