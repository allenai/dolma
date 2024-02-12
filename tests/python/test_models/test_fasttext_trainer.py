import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from fasttext.FastText import _FastText as FastTextModel
from omegaconf import OmegaConf as om

from dolma.models.ft.config import FastTextSupervisedTrainerConfig
from dolma.models.ft.trainer import FastTextTrainer

WORKDIR_PATH = Path(__file__).parent.parent.parent

CONFIG = {
    "model_path": str(WORKDIR_PATH / "work/ft.model"),
    "num_processes": 1,
    "debug": True,
    "model": {"epochs": 50, "learning_rate": 0.2},
    "streams": [
        {
            "documents": [
                WORKDIR_PATH / f"data/topic_classification/{split}.jsonl*",
            ],
            "label": "$.metadata.topics",
            "sample": {
                split: 1.0,
            },
        }
        for split in ["train", "dev", "test"]
    ],
}


class TestFastTextTrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.config = cast(
            FastTextSupervisedTrainerConfig, om.merge(om.structured(FastTextSupervisedTrainerConfig), CONFIG)
        )
        self.stack = ExitStack()
        self.cache_dir = self.stack.enter_context(TemporaryDirectory())
        super().setUp()

    def tearDown(self) -> None:
        if os.path.exists(self.config.model_path):
            os.remove(self.config.model_path)
        self.stack.close()

    def test_config(self):
        self.assertEqual(self.config.model_path, str(WORKDIR_PATH / "work/ft.model"))
        self.assertEqual(self.config.num_processes, 1)
        self.assertTrue(self.config.debug)
        self.assertEqual(self.config.model.learning_rate, CONFIG["model"]["learning_rate"])
        self.assertEqual(self.config.model.epochs, CONFIG["model"]["epochs"])
        self.assertEqual(self.config.model.loss_function, "softmax")
        self.assertEqual(len(self.config.streams), 3)
        self.assertEqual(
            self.config.streams[0].documents[0], str(WORKDIR_PATH / "data/topic_classification/train.jsonl*")
        )

    def test_train(self):
        trainer = FastTextTrainer(self.config, cache_dir=self.cache_dir)
        trainer.do_train()
        self.assertTrue(Path(self.config.model_path).exists())
        model = FastTextModel(trainer.config.model_path)

        labels, _ = model.predict("Battle is the spark that leads to war and combat.")
        self.assertEqual(labels[0], "__label__military")  # type: ignore

        labels, _ = model.predict("sitting down at the dinner table")
        self.assertEqual(labels[0], "__label__food_and_drink")  # type: ignore

        labels, _ = model.predict("commerce and money bolster productivity", k=-1)
        self.assertEqual(labels[0], "__label__economy")  # type: ignore

        self.assertEqual(len(labels), 30)

    def test_valid(self):
        trainer = FastTextTrainer(self.config, cache_dir=self.cache_dir)

        # must train first
        if not os.path.exists(self.config.model_path):
            trainer.do_train()

        self.assertTrue(Path(self.config.model_path).exists())

    def test_predict(self):
        trainer = FastTextTrainer(self.config, cache_dir=self.cache_dir)

        # must train first
        if not os.path.exists(self.config.model_path):
            trainer.do_train()

        trainer.do_test()
        self.assertTrue(Path(self.config.model_path).exists())
