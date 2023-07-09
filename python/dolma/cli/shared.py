from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir

from dolma.cli import field


@dataclass
class WorkDirConfig:
    input: str = field(
        default=str(Path(gettempdir()) / "dolma" / "deduper" / "input"),
        help="Path to the input directory. Required.",
    )
    output: str = field(
        default=str(Path(gettempdir()) / "dolma" / "deduper" / "output"),
        help="Path to the output directory. Required.",
    )
