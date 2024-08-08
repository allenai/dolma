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
    modules: List[str] = field(
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

        for reg_item_name, reg_item_cls in BaseRegistry.registries():
            if parsed_config.filter is not None and parsed_config.filter.lower() not in reg_item_name.lower():
                continue

            any_has_description = any(
                reg_item_desc for _, _, reg_item_desc in reg_item_cls.items_with_description()
            )

            table = Table(title=reg_item_name, style="bold")
            table.width
            table.add_column("name", justify="left", style="cyan", no_wrap=True, ratio=1)
            table.add_column("class", justify="left", style="magenta", no_wrap=False, ratio=1)
            if any_has_description:
                table.add_column("description", justify="left", style="blue", no_wrap=False, ratio=4)

            for reg_item_name, reg_item_cls, reg_item_desc in sorted(reg_item_cls.items_with_description()):
                registry_module = f"{reg_item_cls.__module__}.{reg_item_cls.__name__}"
                if any_has_description:
                    table.add_row(reg_item_name, registry_module, reg_item_desc)
                else:
                    table.add_row(reg_item_name, registry_module)

            console = Console()
            console.print(table)
