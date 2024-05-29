import multiprocessing
import re
import sys
from typing import Callable, List, Optional, TypeVar, Union

import smart_open
from omegaconf.omegaconf import OmegaConf as om

from ..core.paths import cached_path, glob_path
from ..core.registry import BaseRegistry

C = TypeVar("C", bound=Callable)


class ResolverRegistry(BaseRegistry[Callable]):
    @classmethod
    def add(cls, name: str, desc: Optional[str] = None) -> Callable[[C], C]:
        _add_fn = super().add(name, desc)

        def _wrapped_add_fn(
            resolver: C,
            base_add_fn: C = _add_fn,  # type: ignore
            resolver_name: str = name,
        ) -> C:
            base_add_fn(resolver)
            om.register_new_resolver(resolver_name, resolver, replace=True)
            return resolver

        return _wrapped_add_fn


@ResolverRegistry.add("d.cache", "Download a file and replace the path with the cached path.")
def cache(path: str) -> str:
    return str(cached_path(path))


@ResolverRegistry.add("d.glob", "Glob this path and return a list of files.")
def glob(path: str) -> List[str]:
    globbed = list(glob_path(path))
    assert len(globbed) > 0, f"Path {path} does not match any files"
    return globbed


@ResolverRegistry.add("d.procs", "Return the number of processes available (optionally with buffer).")
def processes(n: int = 0) -> int:
    return max(1, multiprocessing.cpu_count() - n)


@ResolverRegistry.add("d.stdin", "Read from stdin and return list of lines.")
def stdin() -> List[str]:
    return [stripped_line for line in sys.stdin if (stripped_line := line.strip())]


@ResolverRegistry.add("d.file", "Read from a file and return contents.")
def file_(path: str, mode: str = "rt", encoding: str = "utf-8") -> str:
    with smart_open.open(path, mode=mode, encoding=encoding) as f:
        return str(f.read())


@ResolverRegistry.add("d.sed", "Perform a sed-like substitution on a string.")
def sed(data: Union[str, List[str]], expression: str, separator: str = "/") -> Union[str, List[str]]:
    if isinstance(data, list):
        return [str(sed(d, expression, separator)) for d in data]

    pattern, replacement = expression.strip(separator).split(separator)
    modified_data = re.sub(pattern, replacement, data)
    return modified_data


@ResolverRegistry.add("d.split", "Split string into list of strings on symbol.")
def split(string: str, symbol: str = "\n") -> List[str]:
    return [stripped_line for line in string.split(symbol) if (stripped_line := line.strip())]
