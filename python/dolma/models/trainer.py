import os
import pickle
from contextlib import ExitStack
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Generic, Type, TypeVar

import smart_open

from ..core.paths import (
    cached_path,
    exists,
    get_cache_dir,
    is_local,
    mkdir_p,
    resource_to_filename,
)
from ..core.utils import dataclass_to_dict
from .config import BaseTrainerConfig, DataConfig, StreamConfig
from .data import BaseDataConverter

T = TypeVar("T", bound=BaseTrainerConfig)


class BaseTrainer(Generic[T]):
    def __init__(self, config: T):
        self.config = config

        if self.config.data is None:
            cache_dir = f"{get_cache_dir()}/{resource_to_filename(pickle.dumps(self.config))}"
            mkdir_p(cache_dir)
            self.config.data = DataConfig(
                train=f"{cache_dir}/train.txt",
                dev=f"{cache_dir}/dev.txt",
                test=f"{cache_dir}/test.txt",
            )
            for stream_config in self.config.streams:
                self.add_stream(stream_config)

    @property
    def data_factory_cls(self) -> Type[BaseDataConverter]:
        raise NotImplementedError("data_factory_cls must be implemented in a subclass")

    def add_stream(self, stream_config: StreamConfig):
        with TemporaryDirectory() as tmpdir:
            assert self.config.data is not None
            self.data_factory_cls.make_stream(output=tmpdir, **dataclass_to_dict(stream_config))
            with ExitStack() as stack:
                current_train_data = stack.enter_context(smart_open.open(self.config.data.train, "at"))
                stream_train_data = stack.enter_context(smart_open.open(f"{tmpdir}/train.txt", "rt"))
                current_train_data.write(stream_train_data.read())

                current_dev_data = stack.enter_context(smart_open.open(self.config.data.dev, "at"))
                stream_dev_data = stack.enter_context(smart_open.open(f"{tmpdir}/dev.txt", "rt"))
                current_dev_data.write(stream_dev_data.read())

                current_test_data = stack.enter_context(smart_open.open(self.config.data.test, "at"))
                stream_test_data = stack.enter_context(smart_open.open(f"{tmpdir}/test.txt", "rt"))
                current_test_data.write(stream_test_data.read())

    def fit(self, data_path: str, save_path: str):
        raise NotImplementedError("train method must be implemented in a subclass")

    def predict(self, data_path: str, load_path: str):
        raise NotImplementedError("valid method must be implemented in a subclass")

    def do_train(self):
        if self.config.data is None or self.config.data.train is None:
            raise ValueError("data.train must be provided")

        if not exists(self.config.data.train):
            raise ValueError(f"data.train {self.config.data.train} does not exist")

        save_path = None
        try:
            if is_local(self.config.model_path):
                save_path = self.config.model_path
            else:
                save_path = (f := NamedTemporaryFile("w", delete=False)).name
                f.close()
            return self.fit(data_path=cached_path(self.config.data.train), save_path=save_path)
        finally:
            if not is_local(self.config.model_path) and save_path and exists(save_path):
                # remote!
                with ExitStack() as stack:
                    fr = stack.enter_context(smart_open.open(save_path, "rb"))
                    fw = stack.enter_context(smart_open.open(self.config.model_path, "wb"))
                    fw.write(fr.read())
                os.remove(save_path)

    def do_valid(self):
        if self.config.data is None or self.config.data.dev is None:
            raise ValueError("data.dev must be provided")

        if not exists(self.config.model_path):
            raise ValueError(f"save_path {self.config.model_path} does not exist")

        return self.predict(data_path=self.config.data.dev, load_path=cached_path(self.config.model_path))

    def do_test(self):
        if self.config.data is None or self.config.data.test is None:
            raise ValueError("data.test must be provided")

        if not exists(self.config.model_path):
            raise ValueError(f"save_path {self.config.model_path} does not exist")

        return self.predict(data_path=self.config.data.test, load_path=cached_path(self.config.model_path))
