"""
Utilities to work with a OmegaConf structured config object

Author: Luca Soldaini (@soldni)
"""


from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import Field
from dataclasses import field as dataclass_field
from dataclasses import is_dataclass
from logging import warn
from typing import Any, Dict, Generic, Literal, Optional, Protocol, Type, TypeVar, Union

from omegaconf import MISSING, DictConfig, ListConfig
from omegaconf import OmegaConf as om
from rich.console import Console
from rich.syntax import Syntax

__all__ = [
    "BaseCli",
    "field",
    "make_parser",
    "namespace_to_nested_omegaconf",
    "print_config",
]


T = TypeVar("T", bound=Any)
D = TypeVar("D", bound="DataClass")
A = TypeVar("A", bound="ArgumentParser")


def _field_nargs(default: Any) -> Union[Literal["?"], Literal["*"]]:
    # return '+' if _default is iterable but not string/bytes, else 1
    if isinstance(default, str) or isinstance(default, bytes):
        return "?"
    elif isinstance(default, Iterable):
        return "*"
    else:
        return "?"


def field(default: T = MISSING, help: Optional[str] = None, **extra: Any) -> T:
    metadata = {"help": help, "type": type(default), "default": default, "nargs": _field_nargs(default), **extra}
    return dataclass_field(default_factory=lambda: deepcopy(default), metadata=metadata)


class DataClass(Protocol):
    __dataclass_fields__: Dict[str, Field]


def make_parser(parser: A, config: Type[DataClass], prefix: Optional[str] = None) -> A:
    for field_name, field in config.__dataclass_fields__.items():
        # get type from annotations or metadata
        typ_ = config.__annotations__.get(field_name, field.metadata.get("type", MISSING))

        if typ_ is MISSING:
            warn(f"No type annotation for field {field_name} in {config.__name__}")
            continue

        if is_dataclass(typ_):
            # recursively add subparsers
            make_parser(parser, typ_, prefix=field_name)
            continue

        field_name = f"{prefix}.{field_name}" if prefix else field_name
        parser.add_argument(
            f"--{field_name}",
            help=field.metadata.get("help"),
            nargs=field.metadata.get("nargs", "?"),
            default=MISSING,
        )

    return parser


def _make_nested_dict(key: str, value: Any, d: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    d = d or {}

    if "." in key:
        key, rest = key.split(".", 1)
        value = _make_nested_dict(rest, value, d.get(key))

    if value is not MISSING:
        d[key] = value

    return d


def namespace_to_nested_omegaconf(args: Namespace, structured: Type[T], config: Optional[dict] = None) -> T:
    nested_config_dict: Dict[str, Any] = {}
    for key, value in vars(args).items():
        nested_config_dict = _make_nested_dict(key, value, nested_config_dict)

    untyped_config: DictConfig = om.merge(
        om.create(config or {}), om.create(nested_config_dict)
    )  # pyright: ignore (pylance is confused because om.create might return a DictConfig or a ListConfig)

    base_structured_config: DictConfig = om.structured(structured)
    merged_config = om.merge(base_structured_config, untyped_config)
    assert isinstance(merged_config, DictConfig)
    return merged_config  # pyright: ignore


def print_config(config: Any, console: Optional[Console] = None) -> None:
    if not isinstance(config, (DictConfig, ListConfig)):
        config = om.create(config)
    console = console or Console()
    syntax = Syntax(code=om.to_yaml(config).strip(), lexer="yaml", theme="ansi_dark")
    console.print(syntax)


class BaseCli(Generic[D]):
    CONFIG: Type[D]
    DESCRIPTION: Optional[str] = None

    @classmethod
    def make_parser(cls, parser: A) -> A:
        assert hasattr(cls, "CONFIG"), f"{cls.__name__} must have a CONFIG attribute"
        return make_parser(parser, cls.CONFIG)

    @classmethod
    def run_from_args(cls, args: Namespace, config: Optional[dict] = None):
        assert hasattr(cls, "CONFIG"), f"{cls.__name__} must have a CONFIG attribute"
        parsed_config = namespace_to_nested_omegaconf(args=args, structured=cls.CONFIG, config=config)
        return cls.run(parsed_config)

    @classmethod
    def run(cls, parsed_config: D):
        raise NotImplementedError("Abstract method; must be implemented in subclass")
