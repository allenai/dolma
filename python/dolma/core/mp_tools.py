import multiprocessing
import time
from contextlib import ExitStack
from multiprocessing.managers import SyncManager
from multiprocessing.pool import Pool
from queue import Queue
from typing import Any, Callable, Dict, Generic, Iterable, Optional, TypeVar, Union

T = TypeVar("T")
R = TypeVar("R")


def get_manager(pool: Union[Pool, "PoolWithDebug"]) -> Union[SyncManager, "ManagerWithDebug"]:
    if getattr(pool, "debug", False):
        return ManagerWithDebug()
    else:
        return multiprocessing.Manager()


class ResultWithDebug(Generic[T]):
    def __init__(self, result: T, *args, **kwargs):
        self.result = result

    def get(self, timeout: Optional[float] = None) -> T:
        return self.result

    def wait(self, timeout: Optional[float] = None) -> None:
        time.sleep(timeout or 0)

    def successful(self) -> bool:
        return True

    def ready(self) -> bool:
        return True


class ManagerWithDebug:
    def Queue(self):
        return Queue()

    def shutdown(self) -> None:
        pass


class PoolWithDebug:
    """A wrapper around multiprocessing.Pool that allows for debugging (i.e., running without multiprocessing).
    Supports creating a manager for shared memory objects (mock in case of debugging)."""

    def __init__(
        self,
        processes: Optional[int] = None,
        initializer: Optional[Callable[..., Any]] = None,
        initargs: Iterable[Any] = (),
        maxtasksperchild: Optional[int] = None,
        debug: bool = False,
    ):
        self.processes = processes
        self.initializer = initializer
        self.initargs = initargs
        self.maxtasksperchild = maxtasksperchild
        self.debug = debug

        # we are gonna keep track of resources in stack; but also keeping them indexed
        # separately for easy access
        self.stack = ExitStack()
        self._manager: Optional[SyncManager] = None
        self._pool: Optional[Pool] = None

        # let's make sure that the start method is spawn for best performance
        try:
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            assert multiprocessing.get_start_method() == "spawn", "Multiprocessing start method must be spawn"

    def __enter__(self):
        if self._pool is None and not self.debug:
            self._pool = self.stack.enter_context(
                Pool(
                    processes=self.processes,
                    initializer=self.initializer,
                    initargs=self.initargs,
                    maxtasksperchild=self.maxtasksperchild,
                )
            )
        return self

    def Manager(self):
        if self._manager is None:
            self._manager = (
                ManagerWithDebug()  # type: ignore
                if self.debug
                else self.stack.enter_context(multiprocessing.Manager())
            )
        return self._manager

    def __exit__(self, *exc):
        return self.stack.close()

    def apply_async(
        self,
        func: Callable[..., R],
        args: Iterable[Any] = (),
        kwds: Dict[str, Any] = {},
        callback: Optional[Callable[[R], Any]] = None,
        error_callback: Optional[Callable[[Any], Any]] = None,
    ):
        if self._pool is None:
            if self.initializer:
                # run the initializer once by calling it with the initargs and then setting it to None
                self.initializer(*self.initargs)
                self.initializer = None
            try:
                resp = func(*args, **kwds)
                if callback is not None:
                    callback(resp)
                return ResultWithDebug(resp)
            except Exception as e:
                if error_callback is not None:
                    error_callback(e)
                raise e
        else:
            return self._pool.apply_async(
                func=func, args=args, kwds=kwds, callback=callback, error_callback=error_callback
            )

    def close(self):
        return self._pool and self._pool.close()

    def join(self):
        return self._pool and self._pool.join()
