import multiprocessing
import time
import warnings
from contextlib import ExitStack
from enum import Enum
from functools import reduce
from hashlib import sha1
from inspect import Parameter
from inspect import signature as get_signature
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type

import tqdm
from rich.progress import filesize
from typing_extensions import TypeAlias, Union

from .loggers import get_logger

if TYPE_CHECKING:
    from .parallel import BaseParallelProcessor


QueueType: TypeAlias = "Queue[Union[None, Tuple[int, ...]]]"


class ServerType(Enum):
    tqdm = "tqdm"
    logger = "logger"
    null = "null"


class BaseProgressBar:
    """One or more progress bars that track progress of a process.

    This class is meant to be subclassed. The subclass must provide one or more attributes of type int, e.g.

    ```python
    class MyProgressBar(BaseProgressBar):
        files: int = 0
        documents: int = 0
    ```

    This class can be used for both adding and running through the progress bars. To start:

    ```python
    queue = Queue()
    pb = MyProgressBar(queue)
    pb.start()

    ... # do some work

    pb.stop()
    ```

    it can also be used in a multiprocessing context:

    ```python
    with Pool(processes=4) as pool:
        queue = mutliprocessing.Manager().Queue()
        pb = MyProgressBar(queue)
        pb.start()

        ... # do some work

        pool.close()
        pool.join()
        pb.stop()
    ```

    If you want to use this class to update a queue:

    ```python
    pb = MyProgressBar(queue)
    pb.files += 1
    pb.documents += 100
    ```
    """

    def __init__(
        self,
        queue: QueueType,
        min_step: int = 1,
        min_time: float = 1e-1,
        server: Union[ServerType, str] = "null",
    ):
        """
        Initialize the ProgressBar object.

        Args:
            queue (QueueType): The queue object to track progress.
            min_step (int, optional): The minimum step size for progress updates. Defaults to 1.
            min_time (float, optional): The minimum time interval between progress updates. Defaults to 1e-1.
            thread (bool, optional): Whether to start the progress bar or use object as client. Defaults to False.
        """
        self._logger = get_logger(self.__class__.__name__, "warn")
        self._queue = queue
        self._last_update_time = time.time()
        self._last_update_step = 0

        self._update_every_seconds = min_time
        self._update_every_steps = min_step

        for field in self.fields():
            setattr(self, field, 0)

        server_mode = ServerType[server] if isinstance(server, str) else server
        if server_mode == ServerType.tqdm:
            self._thread: Optional[Thread] = Thread(
                target=self._run_tqdm,
                kwargs={"queue": queue, "update_every_seconds": min_time, "fields": self.fields()},
                daemon=True,
            )
        elif server_mode == ServerType.logger:
            self._thread = Thread(
                target=self._run_logger,
                kwargs={"queue": queue, "update_every_seconds": min_time, "fields": self.fields()},
                daemon=True,
            )
        else:
            self._thread = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{', '.join(f'{k}={getattr(self, k)}' for k in self.fields())};"
            f" min_step={self._update_every_steps}, min_time={self._update_every_seconds})"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in self.fields() and value > 0:
            self.update()

    @classmethod
    def from_increment_function(cls, processor: "BaseParallelProcessor") -> "Type[BaseProgressBar]":
        # print deprecation warning
        msg = (
            "Deriving progress bar from `increment_progressbar` is deprecated; add a `PROGRESS_BAR_CLS` "
            f"attribute to {type(processor).__name__} instead."
        )
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)

        # checking that the increment_progressbar method is subclassed correctly
        sig = get_signature(processor.increment_progressbar)
        if "queue" not in sig.parameters or sig.parameters["queue"].kind != Parameter.POSITIONAL_ONLY:
            raise AttributeError(
                "increment_progressbar must have a positional-only argument named 'queue'; "
                "Check that you have subclassed BaseParallelProcessor correctly!"
            )
        if "kwargs" in sig.parameters and sig.parameters["kwargs"].kind == Parameter.VAR_KEYWORD:
            raise AttributeError(
                "increment_progressbar must not have a **kwargs argument; "
                "Check that you have subclassed BaseParallelProcessor correctly!"
            )
        if any(p.name != "queue" and p.default != 0 for p in sig.parameters.values()):
            raise AttributeError(
                "increment_progressbar must have a default value of 0 for all arguments except 'queue'; "
                "Check that you have subclassed BaseParallelProcessor correctly!"
            )
        params = [k for k, p in sig.parameters.items() if k != "queue" and p.kind != Parameter.empty]
        h = reduce(lambda h, e: h.update(e.encode()) or h, params, sha1()).hexdigest()  # type: ignore

        # create a new class
        cls_dict = {"__annotations__": {k: int for k in params}, **{p: 0 for p in params}}
        new_cls = type(f"{cls.__name__}{h[-6:]}", (cls,), cls_dict)
        return new_cls

    @classmethod
    def fields(cls) -> Tuple[str, ...]:
        """
        Returns a tuple of field names in the class that are of type int.

        Raises:
            ValueError: If the class does not have at least one field of type int.

        Returns:
            Tuple[str, ...]: A tuple of field names.
        """
        fields: Optional[Tuple[str, ...]] = cls.__dict__.get("__fields__")

        if fields is None:
            fields = tuple(n for n, t in cls.__annotations__.items() if issubclass(t, int))
            setattr(cls, "__fields__", fields)

        if len(fields) == 0:
            raise ValueError(f"Class {cls.__name__} must have at least one field of type int.")

        return fields

    @classmethod
    def parse(cls, values: Optional[Tuple[int, ...]]) -> Dict[str, int]:
        """
        Parses the value from the queue and returns a dictionary mapping field names to their corresponding values.

        Args:
            values (Optional[Tuple[int, ...]]): The values to be parsed for the queue.

        Returns:
            Dict[str, int]: A dictionary mapping field names to their corresponding values.
        """
        if not values:
            return {k: 0 for k in cls.fields()}
        return {k: v for k, v in zip(cls.fields(), values)}

    def _update(self):
        # get the current values
        update = tuple(getattr(self, k, 0) for k in self.fields())

        # time to do an update
        self._queue.put_nowait(update)

        # reset the steps
        self._last_update_step = 0
        self._last_update_time = time.time()

        # reset the steps
        for k in self.fields():
            setattr(self, k, 0)

    def update(self):
        # update the number of steps since the last update
        self._last_update_step += 1

        if self._update_every_steps > self._last_update_step:
            return

        time_before_update = self._last_update_time
        self._update()

        # check if we wanna update frequency based on steps
        if self._queue.qsize() >= multiprocessing.cpu_count():
            self._update_every_steps *= 2
            return

        # check if we wanna update frequency based on time
        if (self._last_update_time - time_before_update) < self._update_every_seconds:
            self._update_every_steps *= 2
            return

    @staticmethod
    def _run_tqdm(queue: QueueType, update_every_seconds: float, fields: Tuple[str, ...]):
        """
        Runs the progress bar.

        This method initializes and updates the progress bars based on the items in the queue.
        It continuously retrieves items from the queue and updates the progress bars accordingly.
        The method exits when a `None` item is retrieved from the queue.

        Returns:
            None
        """
        with ExitStack() as stack:
            pbars = [
                stack.enter_context(tqdm.tqdm(desc=k, unit=k[:1], position=i, unit_scale=True))  # pyright: ignore
                for i, k in enumerate(fields)
            ]

            while True:
                # loop until we get a None
                item = queue.get()
                if item is None:
                    break

                for pbar, value in zip(pbars, item):
                    pbar.update(value)

                time.sleep(update_every_seconds)

    @staticmethod
    def _run_logger(queue: QueueType, update_every_seconds: float, fields: Tuple[str, ...]):
        """
        Run the progress bar update loop.

        Args:
            queue (QueueType): The queue to retrieve items from.
            update_every_seconds (float): The interval between each update in seconds.
            fields (Tuple[str, ...]): The fields to track and display in the progress bar.

        Returns:
            None
        """
        total_counters = {k: 0 for k in fields}
        logger = get_logger("progress", "info")

        while True:
            # loop until we get a None
            item = queue.get()
            if item is None:
                break

            messages = []
            for k, v in zip(fields, item):
                total_counters[k] += v
                unit, suffix = filesize.pick_unit_and_suffix(
                    total_counters[k], ["", "K", "M", "G", "T", "P", "E", "Z", "Y"], 1000
                )
                precision = 1 if suffix else 0
                messages.append(f"{k}: {total_counters[k] / unit:,.{precision}f}{suffix} (+{v:,})")

            logger.info(", ".join(messages))
            time.sleep(update_every_seconds)

    def start(self):
        """Run the progress bar in a separate thread."""
        if self._thread:
            self._thread.start()

    def stop(self):
        """Stop the progress bar.

        This method stops the progress bar by adding a `None` item to the queue and joining the thread.
        """
        self._update()

        if self._thread is not None:
            self._queue.put(None)
            time.sleep(self._update_every_seconds * 2)
            self._thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
