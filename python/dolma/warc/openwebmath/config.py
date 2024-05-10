from dataclasses import dataclass, field
from typing import List, NamedTuple


class ManagerTuple(NamedTuple):
    score: float
    activate: bool


class ThresholdTuple(NamedTuple):
    lo: float
    hi: float


@dataclass
class BoilerPlateConfig:
    ratio_threshold: List[ThresholdTuple] = [ThresholdTuple(0.9, 0.18), ThresholdTuple(0.1, 0.30)]
    absolute_threshold: List[ThresholdTuple] = [ThresholdTuple(0.9, 10), ThresholdTuple(0.1, 20)]
    end_threshold: List[ThresholdTuple] = [ThresholdTuple(0.95, 15), ThresholdTuple(0.05, 5)]
    enable: List[ManagerTuple] = [ManagerTuple(0.95, True), ManagerTuple(0.05, False)]


@dataclass
class TableConfig:
    min_rows: int = 2
    min_cols: int = 3
    format: str = "plain"


@dataclass
class OpenWebMathConfig:
    markdown_headings: List[ManagerTuple] = [ManagerTuple(0.9, True), ManagerTuple(0.1, False)]
    markdown_code: List[ManagerTuple] = [ManagerTuple(0.95, True), ManagerTuple(0.05, False)]
    boilerplate_config: BoilerPlateConfig = field(default_factory=BoilerPlateConfig)
    remove_buttons: bool = True
    remove_image_figures: bool = True
    remove_link_clusters: bool = True
    table_config: TableConfig = field(default_factory=TableConfig)
    remove_chinese: bool = True
    remove_edit_buttons: bool = True
    extract_latex: bool = True
