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
from logging import warning
from typing import (
    Any,
    Dict,
    Generic,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from omegaconf import MISSING, DictConfig, ListConfig
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from rich.console import Console
from rich.syntax import Syntax

from ..core.errors import DolmaConfigError

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
    if isinstance(default, (str, bytes)):
        return "?"

    if isinstance(default, Iterable):
        return "*"

    return "?"


def field(default: T = MISSING, help: Optional[str] = None, **extra: Any) -> T:
    metadata = {"help": help, "type": type(default), "default": default, "nargs": _field_nargs(default), **extra}
    return dataclass_field(default_factory=lambda: deepcopy(default), metadata=metadata)


class DataClass(Protocol):
    __dataclass_fields__: Dict[str, Field]


def make_parser(parser: A, config: Type[DataClass], prefix: Optional[str] = None) -> A:
    for field_name, dt_field in config.__dataclass_fields__.items():
        # get type from annotations or metadata
        typ_ = config.__annotations__.get(field_name, dt_field.metadata.get("type", MISSING))

        if typ_ is MISSING:
            warning(f"No type annotation for field {field_name} in {config.__name__}")
            continue

        # join prefix and field name
        field_name = f"{prefix}.{field_name}" if prefix else field_name

        # This section here is to handle Optional[T] types; we only care for cases where T is a dataclass
        # So we first check if type is Union since Optional[T] is just a shorthand for Union[T, None]
        # and that the union contains only one non-None type
        if get_origin(typ_) == Union:
            # get all non-None types
            args = [a for a in get_args(typ_) if a is not type(None)]  # noqa: E721

            if len(args) == 1:
                # simple Optional[T] type
                typ_ = args[0]

        # here's where we check if T is a dataclass
        if is_dataclass(typ_):
            # recursively add subparsers
            make_parser(parser, typ_, prefix=field_name)  # type: ignore
            continue

        if typ_ is bool:
            # for boolean values, we add two arguments: --field_name and --no-field_name
            parser.add_argument(
                f"--{field_name}",
                help=dt_field.metadata.get("help"),
                dest=field_name,
                action="store_true",
                default=MISSING,
            )
            parser.add_argument(
                f"--no-{field_name}",
                help=f"Disable {field_name}",
                dest=field_name,
                action="store_false",
                default=MISSING,
            )
        else:
            # else it's just a normal argument
            parser.add_argument(
                f"--{field_name}",
                help=dt_field.metadata.get("help"),
                nargs=dt_field.metadata.get("nargs", "?"),
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

    # check for type
    if not isinstance(merged_config, DictConfig):
        raise DolmaConfigError(f"Expected a DictConfig, got {type(merged_config).__name__}")

    # try resolving all cross references in the config, raise a DolmaConfigError if it fails
    try:
        om.resolve(merged_config)
    except OmegaConfBaseException as ex:
        raise DolmaConfigError(f"Invalid error while parsing key `{ex.full_key}`: {type(ex).__name__}") from ex

    return merged_config  # pyright: ignore


def print_config(config: Any, console: Optional[Console] = None) -> None:
    if not isinstance(config, (DictConfig, ListConfig)):
        config = om.create(config)

    # print the config as yaml using a rich syntax highlighter
    console = console or Console()
    yaml_config = om.to_yaml(config, sort_keys=True).strip()
    highlighted = Syntax(code=yaml_config, lexer="yaml", theme="ansi_dark")
    console.print(highlighted)


class BaseCli(Generic[D]):
    CONFIG: Type[D]
    DESCRIPTION: Optional[str] = None

    @classmethod
    def make_parser(cls, parser: A) -> A:
        assert hasattr(cls, "CONFIG"), f"{cls.__name__} must have a CONFIG attribute"
        return make_parser(parser, cls.CONFIG)  # pyright: ignore

    @classmethod
    def run_from_args(cls, args: Namespace, config: Optional[dict] = None):
        assert hasattr(cls, "CONFIG"), f"{cls.__name__} must have a CONFIG attribute"
        parsed_config = namespace_to_nested_omegaconf(
            args=args, structured=cls.CONFIG, config=config  # pyright: ignore
        )
        try:
            return cls.run(parsed_config)
        except OmegaConfBaseException as ex:
            raise DolmaConfigError(
                f"Invalid error while parsing key `{ex.full_key}` of `{ex.object_type_str}`: "
                f"{type(ex).__name__}"
            ) from ex

    @classmethod
    def run(cls, parsed_config: D):
        raise NotImplementedError("Abstract method; must be implemented in subclass")
