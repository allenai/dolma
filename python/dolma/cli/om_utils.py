'''
Utilities to work with a OmegaConf structured config object

Author: Luca Soldaini (@soldni)
'''

from argparse import ArgumentParser, Namespace
from logging import warn
from collections.abc import Iterable
from omegaconf import DictConfig, OmegaConf as om, MISSING
from dataclasses import Field, is_dataclass, field as dataclass_field
from copy import deepcopy

from typing import Any, Literal, Optional, Protocol, Dict, TypeVar, Union


__all__ = [
    'field',
    'make_parser',
    'namespace_to_nested_omegaconf'
]


T = TypeVar('T', bound=Any)


def _field_nargs(default: Any) -> Union[Literal['?'], Literal['*']]:
    # return '+' if _default is iterable but not string/bytes, else 1
    if isinstance(default, str) or isinstance(default, bytes):
        return '?'
    elif isinstance(default, Iterable):
        return '*'
    else:
        return '?'


def field(
    default: T = MISSING,
    help: Optional[str] = None,
    **extra: Any
) -> T:
    metadata = {
        'help': help,
        'type': type(default),
        'default': default,
        'nargs': _field_nargs(default),
        **extra
    }
    return dataclass_field(default_factory=lambda: deepcopy(default), metadata=metadata)


class DataClass(Protocol):
    __dataclass_fields__: Dict[str, Field]


def make_parser(parser: ArgumentParser, config: DataClass, prefix: Optional[str] = None) -> ArgumentParser:
    for field_name, field in config.__dataclass_fields__.items():
        # get type from annotations or metadata
        typ_ = config.__annotations__.get(field_name, field.metadata.get('type', MISSING))

        if typ_ is MISSING:
            warn(f'No type annotation for field {field_name} in {config.__class__.__name__}')
            continue

        if is_dataclass(typ_):
            # recursively add subparsers
            make_parser(parser, typ_, prefix=field_name)
            continue

        field_name = f'{prefix}.{field_name}' if prefix else field_name
        parser.add_argument(f'--{field_name}', **{**field.metadata, 'type': typ_})

    return parser


def _make_nested_dict(key: str, value: Any, d: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    d = d or {}
    if '.' in key:
        key, rest = key.split('.', 1)
        value = _make_nested_dict(rest, value, d.get(key))
    d[key] = value
    return d


def namespace_to_nested_omegaconf(namespace: Namespace, dt: DataClass) -> DictConfig:
    nested_config_dict: Dict[str, Any] = {}
    for key, value in namespace.__dict__.items():
        nested_config_dict = _make_nested_dict(key, value, nested_config_dict)
    untyped_config: DictConfig = om.create(nested_config_dict)
    base_structured_config: DictConfig = om.structured(dt)
    merged_config = om.merge(base_structured_config, untyped_config)
    assert isinstance(merged_config, DictConfig)
    return merged_config
