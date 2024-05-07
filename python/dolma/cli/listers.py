from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from dolma.cli import BaseCli, field
from dolma.core.loggers import get_logger
from dolma.core.registry import BaseRegistry
from dolma.core.utils import import_modules


@dataclass
class ListerConfig:
    modules : List[str] = field(
        default=[],
        help="List of Python modules $PYTHONPATH to import custom registry modules from.",
    )
    tagger_modules: Optional[List[str]] = field(
        default=None,
        help="List of Python modules $PYTHONPATH to import custom taggers from.",
    )
    filter: Optional[str] = field(
        default=None,
        help="Filter which registries to list.",
    )


class ListerCli(BaseCli):
    CONFIG = ListerConfig
    DESCRIPTION = "List all available modules in registry."

    @classmethod
    def run(cls, parsed_config: ListerConfig):
        if parsed_config.tagger_modules is not None:
            # deprecation warning
            logger = get_logger(__file__)
            logger.warning(
                "The `tagger_modules` argument is deprecated and will be removed in a future release. "
                "Please use `modules` instead."
            )
            parsed_config.modules.extend(parsed_config.tagger_modules)

        # import tagger modules
        import_modules(parsed_config.modules)

        for tagger_name, tagger_cls in BaseRegistry.registries():
            if parsed_config.filter is not None and parsed_config.filter.lower() not in tagger_name.lower():
                continue

            table = Table(title=tagger_name, style="bold")
            table.add_column("name", justify="left", style="cyan")
            table.add_column("class", justify="left", style="magenta")

            for tagger_name, tagger_cls in sorted(tagger_cls.items()):
                tagger_repr = f"{tagger_cls.__module__}.{tagger_cls.__name__}"
                table.add_row(tagger_name, tagger_repr)

            console = Console()
            console.print(table)
