from typing import Callable, Dict
from .base import Task

__all__ = [
    "task_registry",
    "register_task",
]

class TaskRegistry:
    _instance = None
    _tasks: Dict[str, Callable[[], Task]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskRegistry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str | None = None):
        def decorator(func: Callable[[], Task]):
            cls._tasks[name or func.__name__] = func
            return func
        return decorator

    @classmethod
    def get_task(cls, name: str) -> Callable[[], Task]:
        return cls._tasks.get(name)

    @classmethod
    def list_tasks(cls) -> list[str]:
        return list(cls._tasks.keys())


task_registry = TaskRegistry()
register_task = task_registry.register
