import os
import pickle
from contextlib import ExitStack
from hashlib import sha256
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Generic, Optional, Type, TypeVar

import smart_open

from ..core.paths import cached_path, exists, get_cache_dir, is_local, mkdir_p
from .config import BaseTrainerConfig, DataConfig, StreamConfig
from .data import BaseDataConverter

T = TypeVar("T", bound=BaseTrainerConfig)


class BaseTrainer(Generic[T]):
    def __init__(self, config: T, cache_dir: Optional[str] = None):
        self.config = config

        if self.config.data is None:
            # No data provided in the format expected by the model, so we derive from
            # streams in dolma format. Let's first check if any stream is provided
            assert self.config.streams is not None, "streams must be provided if data is not provided"

            # the fingerprint of the streams is used to create a unique cache directory
            streams_fingerprint = sha256(pickle.dumps(self.config.streams)).hexdigest()
            cache_dir = cache_dir or f"{get_cache_dir()}/{streams_fingerprint}"
            self.config.data = DataConfig(
                train=f"{cache_dir}/train.txt",
                dev=f"{cache_dir}/dev.txt",
                test=f"{cache_dir}/test.txt",
            )

            # let's check if the cache directory exists and if the files are there; if
            # so, we return immediately bc we don't need to create the files again
            if all(
                fp is not None and exists(fp)
                for fp in (self.config.data.train, self.config.data.dev, self.config.data.test)
            ):
                return

            # they do not exist, so we create them by adding the streams one by one.
            mkdir_p(cache_dir)
            for stream_config in self.config.streams:
                self.add_stream(stream_config)

    @property
    def data_factory_cls(self) -> Type[BaseDataConverter]:
        """This property returns the data factory class that will be used to process
        streams into the format expected by the model. It must be implemented in a
        subclass."""
        raise NotImplementedError("data_factory_cls must be implemented in a subclass")

    def add_stream(self, stream_config: StreamConfig):
        """Create data files from a stream, and merge them with the current data files."""

        with TemporaryDirectory() as tmpdir:
            # we need to make sure that the data is not None, otherwise we cannot merge the streams
            assert self.config.data is not None

            # we use the data factory transform the stream into the format expected by the model.
            # note that the new stream is created in a temporary directory, to be merged with the
            # current data files later.
            self.data_factory_cls.make_stream(
                output=tmpdir,
                documents=stream_config.documents,
                word_tokenizer=self.config.word_tokenizer,
                text_selector=stream_config.text,
                label_selector=stream_config.label,
                train_sample_rate=stream_config.sample.train,
                dev_sample_rate=stream_config.sample.dev,
                test_sample_rate=stream_config.sample.test,
                debug=self.config.debug,
                num_processes=self.config.num_processes,
            )
            # this is where we merge the streams with the current data files
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

    def fit(self, data_path: str, save_path: str, validation_path: Optional[str] = None):
        raise NotImplementedError("train method must be implemented in a subclass")

    def predict(self, data_path: str, load_path: str):
        raise NotImplementedError("valid method must be implemented in a subclass")

    def do_train(self):
        if self.config.data is None or self.config.data.train is None:
            raise ValueError("data.train must be provided")

        if not exists(self.config.data.train):
            raise ValueError(f"data.train {self.config.data.train} does not exist")

        if self.config.data.dev is not None and not exists(self.config.data.dev):
            validation_path = cached_path(self.config.data.dev)
        else:
            validation_path = None

        save_path = None
        try:
            if is_local(self.config.model_path):
                save_path = self.config.model_path
            else:
                save_path = (f := NamedTemporaryFile("w", delete=False)).name
                f.close()
            fit = self.fit(
                data_path=cached_path(self.config.data.train), save_path=save_path, validation_path=validation_path
            )
        finally:
            if not is_local(self.config.model_path) and save_path and exists(save_path):
                # remote!
                with ExitStack() as stack:
                    fr = stack.enter_context(smart_open.open(save_path, "rb"))
                    fw = stack.enter_context(smart_open.open(self.config.model_path, "wb"))
                    fw.write(fr.read())
                os.remove(save_path)
        return fit

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
