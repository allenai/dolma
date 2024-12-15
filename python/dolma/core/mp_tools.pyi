from collections.abc import Callable, Iterable
from multiprocessing.managers import SyncManager
from multiprocessing.pool import ApplyResult, Pool
from typing import Any

class ResultWithDebug(ApplyResult): ...  # noqa: E701,E302
class ManagerWithDebug(SyncManager): ...  # noqa: E701

class PoolWithDebug(Pool):  # noqa: E302
    def __init__(  # noqa: E704
        self,
        processes: int | None = None,
        initializer: Callable[..., Any] | None = None,
        initargs: Iterable[Any] = (),
        maxtasksperchild: int | None = None,
        debug: bool = False,
    ): ...

def get_manager(pool: Pool) -> SyncManager: ...  # noqa: E701, E704, E302
