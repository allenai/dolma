import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from fasttext.FastText import _FastText as FastTextModel
from omegaconf import OmegaConf as om

from dolma.models.fasttext_model.config import FastTextSupervisedTrainerConfig
from dolma.models.fasttext_model.trainer import FastTextTrainer
from dolma.models.word_tokenizers import TokenizerRegistry

WORKDIR_PATH = Path(__file__).parent.parent.parent

SUPERVISED_CONFIG = {
    "model_path": str(WORKDIR_PATH / "work/supervised_ft.model"),
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

UNSUPERVISED_CONFIG = (
    {**SUPERVISED_CONFIG, "model": {"epochs": 10, "algorithm": "skipgram", "learning_rate": 0.1}},
)


class TestFastTextTrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.config = cast(
            FastTextSupervisedTrainerConfig,
            om.merge(om.structured(FastTextSupervisedTrainerConfig), SUPERVISED_CONFIG),
        )
        self.stack = ExitStack()
        self.config.cache_dir = self.stack.enter_context(TemporaryDirectory())
        super().setUp()

    def tearDown(self) -> None:
        if os.path.exists(self.config.model_path):
            os.remove(self.config.model_path)
        self.stack.close()

    def test_config(self):
        self.assertEqual(self.config.model_path, str(WORKDIR_PATH / "work/supervised_ft.model"))
        self.assertEqual(self.config.num_processes, 1)
        self.assertTrue(self.config.debug)
        self.assertEqual(self.config.model.learning_rate, SUPERVISED_CONFIG["model"]["learning_rate"])
        self.assertEqual(self.config.model.epochs, SUPERVISED_CONFIG["model"]["epochs"])
        self.assertEqual(self.config.model.loss_function, "softmax")
        self.assertEqual(len(self.config.streams), 3)
        self.assertEqual(
            self.config.streams[0].documents[0], str(WORKDIR_PATH / "data/topic_classification/train.jsonl*")
        )
        self.assertEqual(self.config.word_tokenizer, "punct")

    def test_train(self):
        trainer = FastTextTrainer(self.config)
        trainer.do_train()
        self.assertTrue(Path(self.config.model_path).exists())
        model = FastTextModel(trainer.config.model_path)
        tokenizer = TokenizerRegistry.get(trainer.config.word_tokenizer)()

        labels, _ = model.predict(next(tokenizer("Battle is the spark that leads to war and combat.")))
        self.assertEqual(labels[0], "__label__military")  # pyright: ignore

        labels, _ = model.predict(next(tokenizer("sitting down at the dinner table")))
        self.assertEqual(labels[0], "__label__food_and_drink")  # pyright: ignore

        labels, _ = model.predict(next(tokenizer("commerce and money bolster productivity")), k=-1)
        self.assertEqual(labels[0], "__label__business")  # pyright: ignore

        self.assertEqual(len(labels), 30)

    def test_valid(self):
        trainer = FastTextTrainer(self.config)

        # must train first
        if not os.path.exists(self.config.model_path):
            trainer.do_train()

        self.assertTrue(Path(self.config.model_path).exists())

    def test_predict(self):
        trainer = FastTextTrainer(self.config)

        # must train first
        if not os.path.exists(self.config.model_path):
            trainer.do_train()

        trainer.do_test()
        self.assertTrue(Path(self.config.model_path).exists())


# class TestFastTextUnsupervisedTrainer(unittest.TestCase):
#     def setUp(self) -> None:
#         self.config = cast(
#             FastTextSupervisedTrainerConfig,
#             om.merge(om.structured(FastTextSupervisedTrainerConfig), SUPERVISED_CONFIG),
#         )
#         self.stack = ExitStack()
#         self.cache_dir = self.stack.enter_context(TemporaryDirectory())
#         super().setUp()

#     def tearDown(self) -> None:
#         if os.path.exists(self.config.model_path):
#             os.remove(self.config.model_path)
#         self.stack.close()

#     def test_train(self):
#         trainer = FastTextTrainer(self.config)
#         trainer.do_train()
#         self.assertTrue(Path(self.config.model_path).exists())
#         model = FastTextModel(trainer.config.model_path)

#         # TODO: finish this test
#         self.assertTrue(False)
