from dataclasses import Field
from omegaconf import omegaconf as om
from typing import Callable, Dict, Generator, NamedTuple, Optional, Protocol, Any, Tuple, TypeVar
from argparse import ArgumentParser

from typing_extensions import TypeAlias


class ConfigType(Protocol):
    __dataclass_fields__: Dict[str, Any]


EntrypointFn: TypeAlias = Callable[[ConfigType], None]
EF = TypeVar("EF", bound=EntrypointFn)


class Cli:
    _entrypoints: Dict[str, EntrypointFn] = {}

    def _add_entrypoint(self, name: str, fn: EF) -> EF:
        self._entrypoints[name] = fn
        return fn

    def _get_all_config_fields(
        self,
        dt: ConfigType,
        base_name: Optional[str] = None
    ) -> Generator[Tuple[str, Field], None, None]:
        assert '__dataclass_fields__' in dir(dt), f'Expected {dt} to be a dataclass.'

        for name, field in sorted(dt.__dataclass_fields__.items()):
            if base_name is None:
                yield name, field
            else:
                yield f'{base_name}.{name}', field

    def add(self, name: str) -> Callable[[EF], EF]:
        """Decorator to add an entrypoint to the CLI.

        An entrypoint is a function that takes a single argument as input (a config dataclass object)
        and runs a command.

        Args:
            name (str): The name of the entrypoint.
        """
        def decorator(fn: EF) -> EF:
            return self._add_entrypoint(name, fn)

        return decorator

    def run(self):
        ap = ArgumentParser()
