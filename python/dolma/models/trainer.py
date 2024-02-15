import os
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import Generic, List, Optional, Type, TypeVar, Union

import smart_open

from ..core.paths import (
    cached_path,
    exists,
    get_cache_dir,
    is_local,
    join_path,
    mkdir_p,
    parent,
)
from ..core.utils import make_fingerprint
from .config import BaseTrainerConfig, DataConfig
from .data import BaseDataConverter, combine_splits

T = TypeVar("T", bound=BaseTrainerConfig)

COMBINED_STREAMS = "_COMBINED_STREAMS"
HASH_CHARS = 12


def already_processed(data: DataConfig, override: bool = False) -> bool:
    """Check if the data has already been processed."""
    if override:
        return False
    return all(fp is not None and exists(fp) for fp in (data.train, data.dev, data.test))


class BaseTrainer(Generic[T]):
    def __init__(self, config: T):
        self.config = config

        if self.config.data is not None:
            return

        # No data provided in the format expected by the model, so we derive from
        # streams in dolma format. Let's first check if any stream is provided
        assert self.config.streams is not None, "streams must be provided if data is not provided"

        # the fingerprint of the streams is used to create a unique cache directory (make sure it exists!)
        streams_fingerprint = make_fingerprint(self.config.streams, self.config.word_tokenizer)
        base_data_dir = self.config.cache_dir or join_path("", get_cache_dir(), streams_fingerprint)
        mkdir_p(base_data_dir)

        # we create the data configuration from the cache directory
        self.config.data = DataConfig.from_dir(
            combined_dir := join_path("", base_data_dir, f"{COMBINED_STREAMS}_{streams_fingerprint[:HASH_CHARS]}")
        )

        # let's check if the cache directory exists and if the files are there; if
        # so, we return immediately bc we don't need to create the files again
        if already_processed(self.config.data, override=self.config.reprocess_streams):
            return

        processor: Union[None, BaseDataConverter] = None
        stream_output_dirs: List[str] = []
        for stream_config in self.config.streams:
            assert stream_config.name != COMBINED_STREAMS, f"stream name {COMBINED_STREAMS} is reserved"

            # this is the name of the directory where data from this stream will be stored
            # we combine the fingerprint of the stream with an optional name field if the name
            # field is provided.
            single_stream_fingerprint = make_fingerprint(stream_config, self.config.word_tokenizer)
            if stream_config.name is not None:
                single_stream_fingerprint = f"{stream_config.name}_{single_stream_fingerprint[:HASH_CHARS]}"

            # this is where the stream will be stored.
            stream_dir = join_path("", base_data_dir, single_stream_fingerprint)
            stream_output_dirs.append(stream_dir)
            if already_processed(DataConfig.from_dir(stream_dir), override=self.config.reprocess_streams):
                continue

            stream_processor = self.data_factory_cls.make(
                output=stream_dir,
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
            processor = (processor + stream_processor) if processor is not None else stream_processor

        if processor is not None:
            # some splits have to be processed!
            processor()

        # merge the splits into a single directory
        combine_splits(sources=stream_output_dirs, destination=combined_dir)

    @property
    def data_factory_cls(self) -> Type[BaseDataConverter]:
        """This property returns the data factory class that will be used to process
        streams into the format expected by the model. It must be implemented in a
        subclass."""
        raise NotImplementedError("data_factory_cls must be implemented in a subclass")

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

        local_save_path: Optional[str] = None
        try:
            # if the model path is local, we can save directly to it
            if is_local(self.config.model_path):
                local_save_path = self.config.model_path
            else:
                local_save_path = (f := NamedTemporaryFile("w", delete=False)).name
                f.close()

            # make sure the parent directory exists
            mkdir_p(parent(local_save_path))

            # actually train the model
            fit = self.fit(
                data_path=cached_path(self.config.data.train),
                save_path=local_save_path,
                validation_path=validation_path,
            )
        finally:
            if not is_local(self.config.model_path) and local_save_path and exists(local_save_path):
                # remote!
                with ExitStack() as stack:
                    fr = stack.enter_context(smart_open.open(local_save_path, "rb"))
                    fw = stack.enter_context(smart_open.open(self.config.model_path, "wb"))
                    fw.write(fr.read())
                os.remove(local_save_path)
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
