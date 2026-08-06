"""Hierarchical source, target, and materialized-output reporting."""

from __future__ import annotations

import html
from collections import Counter, defaultdict
from typing import Any, Sequence


def _human_count(value: int) -> str:
    for scale, suffix in (
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:,}"


def _multiplier(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}×"
    if value >= 10:
        return f"{value:.1f}×"
    return f"{value:.2f}×"


def _status(statuses_by_leaf: dict[str, list[str]], leaves: Sequence[str]) -> str:
    statuses = [status for leaf in leaves for status in statuses_by_leaf.get(leaf, [])]
    if not statuses:
        return "dropped"
    return "passed" if all(status == "passed" for status in statuses) else "outside_tolerance"


def _node(
    name: str,
    source: int,
    target: int,
    actual: int | None,
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "target": target,
        "actual": actual,
        "status": status,
        **extra,
    }


def _csv_row(
    *,
    level: str,
    source_family: str,
    source: int,
    target: int,
    actual: int | None,
    status: str,
    subcategory_name: str = "",
    category_name: str = "",
    lower_group: str = "",
    leaf_id: str = "",
) -> dict[str, Any]:
    return {
        "level": level,
        "source_family": source_family,
        "subcategory_name": subcategory_name,
        "category_name": category_name,
        "lower_group": lower_group,
        "leaf_id": leaf_id,
        "source_uint32_values": source,
        "target_uint32_values": target,
        "actual_uint32_values": "" if actual is None else actual,
        "actual_minus_target": "" if actual is None else actual - target,
        "planned_sampling_ratio": "" if source == 0 else f"{target / source:.12g}",
        "realized_sampling_ratio": ("" if actual is None or source == 0 else f"{actual / source:.12g}"),
        "status": status,
    }


def _build_hierarchy(
    inventory_details: dict[str, Any],
    validation_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    actual_by_leaf: dict[str, int] = defaultdict(int)
    unit_count_by_leaf: Counter[str] = Counter()
    shard_count_by_leaf: Counter[str] = Counter()
    statuses_by_leaf: dict[str, list[str]] = defaultdict(list)
    for row in validation_rows:
        leaf_id = str(row["leaf_id"])
        actual_by_leaf[leaf_id] += int(row["actual_uint32_values"])
        unit_count_by_leaf[leaf_id] += 1
        shard_count_by_leaf[leaf_id] += int(row["npy_count"])
        statuses_by_leaf[leaf_id].append(str(row["status"]))
    selected_leaf_ids = set(actual_by_leaf)
    selected_mix_names = {str(row["mix_name"]) for row in validation_rows}
    flat_rows: list[dict[str, Any]] = []
    subcategories: list[dict[str, Any]] = []

    for source_row in inventory_details.get("sources", []):
        mix_name = str(source_row["mix_name"])
        if mix_name not in selected_mix_names:
            continue
        family_name = str(source_row["source_family"])
        subcategory_name = str(source_row["subcategory_name"])
        categories: list[dict[str, Any]] = []
        for category_row in source_row.get("categories", []):
            leaf_id = str(category_row["leaf_id"])
            source = int(category_row["source_uint32_values"])
            target = int(category_row["target_uint32_values"])
            if leaf_id not in selected_leaf_ids and target > 0:
                continue
            actual = actual_by_leaf.get(leaf_id, 0)
            status = _status(statuses_by_leaf, [leaf_id]) if leaf_id in selected_leaf_ids else "dropped"
            lower_groups: list[dict[str, Any]] = []
            for lower_row in category_row.get("lower_groups", []):
                lower_source = int(lower_row["source_uint32_values"])
                lower_target = int(lower_row["implied_target_uint32_values"])
                lower_actual = 0 if status == "dropped" else None
                lower = _node(
                    str(lower_row["lower_group"]),
                    lower_source,
                    lower_target,
                    lower_actual,
                    "dropped" if lower_actual == 0 else "planned_only",
                    npy_count=int(lower_row["unique_npy_count"]),
                )
                lower_groups.append(lower)
                flat_rows.append(
                    _csv_row(
                        level="lower_group",
                        source_family=family_name,
                        subcategory_name=subcategory_name,
                        category_name=str(category_row["category_name"]),
                        lower_group=lower["name"],
                        leaf_id=leaf_id,
                        source=lower_source,
                        target=lower_target,
                        actual=lower_actual,
                        status=lower["status"],
                    )
                )
            category = _node(
                str(category_row["category_name"]),
                source,
                target,
                actual,
                status,
                leaf_id=leaf_id,
                unit_count=unit_count_by_leaf[leaf_id],
                output_shards=shard_count_by_leaf[leaf_id],
                lower_groups=sorted(
                    lower_groups,
                    key=lambda row: (row["target"], row["source"], row["name"]),
                    reverse=True,
                ),
            )
            categories.append(category)
            flat_rows.append(
                _csv_row(
                    level="category",
                    source_family=family_name,
                    subcategory_name=subcategory_name,
                    category_name=category["name"],
                    leaf_id=leaf_id,
                    source=source,
                    target=target,
                    actual=actual,
                    status=status,
                )
            )
        if not categories:
            continue
        source = sum(row["source"] for row in categories)
        target = sum(row["target"] for row in categories)
        actual = sum(row["actual"] for row in categories)
        leaves = [row["leaf_id"] for row in categories]
        status = _status(statuses_by_leaf, leaves)
        subcategory = _node(
            subcategory_name,
            source,
            target,
            actual,
            status,
            source_family=family_name,
            mix_name=mix_name,
            leaf_ids=leaves,
            categories=sorted(
                categories,
                key=lambda row: (row["target"], row["source"], row["name"]),
                reverse=True,
            ),
        )
        subcategories.append(subcategory)
        flat_rows.append(
            _csv_row(
                level="subcategory",
                source_family=family_name,
                subcategory_name=subcategory_name,
                source=source,
                target=target,
                actual=actual,
                status=status,
            )
        )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for subcategory in subcategories:
        by_family[subcategory["source_family"]].append(subcategory)
    families: list[dict[str, Any]] = []
    for family_name, children in by_family.items():
        source = sum(row["source"] for row in children)
        target = sum(row["target"] for row in children)
        actual = sum(row["actual"] for row in children)
        leaves = [leaf for row in children for leaf in row["leaf_ids"]]
        status = _status(statuses_by_leaf, leaves)
        family = _node(
            family_name,
            source,
            target,
            actual,
            status,
            subcategories=sorted(
                children,
                key=lambda row: (row["target"], row["source"], row["name"]),
                reverse=True,
            ),
        )
        families.append(family)
        flat_rows.append(
            _csv_row(
                level="source_family",
                source_family=family_name,
                source=source,
                target=target,
                actual=actual,
                status=status,
            )
        )
    families.sort(
        key=lambda row: (row["target"], row["source"], row["name"]),
        reverse=True,
    )
    flat_rows.sort(
        key=lambda row: (
            row["source_family"],
            row["subcategory_name"],
            row["category_name"],
            row["lower_group"],
            row["level"],
        )
    )
    return families, flat_rows


def _metric(label: str, value: str, warning: bool = False) -> str:
    warning_class = " validation-warning" if warning else ""
    return (
        f'<span class="validation-metric{warning_class}">'
        f'<span class="validation-label">{html.escape(label)}</span>'
        f'<span class="validation-value">{html.escape(value)}</span></span>'
    )


def _sampling_label(source: int, target: int, actual: int | None) -> str:
    if source <= 0:
        return "—"
    planned = _multiplier(target / source)
    if actual is None:
        return planned
    return f"{planned} planned · {_multiplier(actual / source)} actual"


def _delta_label(target: int, actual: int) -> str:
    delta = actual - target
    sign = "+" if delta > 0 else ""
    percent = 100 * delta / target if target else 0.0
    return f"{sign}{_human_count(delta)} · {percent:+.5f}%"


def _bars(target: int, actual: int | None, maximum: int) -> str:
    target_width = 100 * target / maximum if maximum else 0.0
    actual_width = 100 * actual / maximum if actual is not None and maximum else 0.0
    actual_bar = (
        '<span class="actual-track"><span class="actual-fill" ' f'style="width:{actual_width:.8f}%"></span></span>'
        if actual is not None
        else ""
    )
    return (
        '<span class="target-actual-bars" aria-hidden="true">'
        '<span class="target-track"><span class="target-fill" '
        f'style="width:{target_width:.8f}%"></span></span>'
        f"{actual_bar}</span>"
    )


def _values(node: dict[str, Any], maximum: int) -> str:
    actual = node["actual"]
    return (
        _metric("Source", f"{_human_count(node['source'])} tokens")
        + _metric("Target", f"{_human_count(node['target'])} tokens")
        + _metric(
            "Actual",
            "—" if actual is None else f"{_human_count(actual)} tokens",
        )
        + _metric(
            "Sampling",
            _sampling_label(node["source"], node["target"], actual),
        )
        + (
            _metric(
                "Actual − target",
                _delta_label(node["target"], actual),
                warning=node["status"] == "outside_tolerance",
            )
            if actual is not None
            else _metric("Actual − target", "—")
        )
        + _bars(node["target"], actual, maximum)
    )


def _render_hierarchy(families: Sequence[dict[str, Any]]) -> str:
    family_maximum = max(
        (max(row["target"], row["actual"]) for row in families),
        default=1,
    )
    family_sections: list[str] = []
    for family in families:
        subcategory_maximum = max(
            (max(row["target"], row["actual"]) for row in family["subcategories"]),
            default=1,
        )
        subcategory_sections: list[str] = []
        for subcategory in family["subcategories"]:
            category_maximum = max(
                (max(row["target"], row["actual"]) for row in subcategory["categories"]),
                default=1,
            )
            category_sections: list[str] = []
            for category in subcategory["categories"]:
                lower_maximum = max(
                    (row["target"] for row in category["lower_groups"]),
                    default=1,
                )
                lower_rows = "".join(
                    '<div class="validation-row lower-group-row">'
                    f'<span class="validation-name">{html.escape(lower["name"])}'
                    f'<span class="validation-context">{lower["npy_count"]:,} NPY</span></span>'
                    + _values(lower, lower_maximum)
                    + "</div>"
                    for lower in category["lower_groups"]
                )
                context = (
                    "not selected by the sampling plan"
                    if category["status"] == "dropped"
                    else (
                        f"{category['unit_count']:,} unit"
                        f"{'s' if category['unit_count'] != 1 else ''} · "
                        f"{category['output_shards']:,} output shard"
                        f"{'s' if category['output_shards'] != 1 else ''}"
                    )
                )
                category_sections.append(
                    '<details class="validation-node category-node">'
                    '<summary class="validation-row">'
                    f'<span class="validation-name">{html.escape(category["name"])}'
                    f'<span class="validation-context">{html.escape(context)}</span></span>'
                    + _values(category, category_maximum)
                    + "</summary>"
                    + (
                        '<div class="validation-children lower-groups">' + lower_rows + "</div>"
                        if lower_rows
                        else ""
                    )
                    + "</details>"
                )
            subcategory_sections.append(
                '<details class="validation-node subcategory-node">'
                '<summary class="validation-row">'
                f'<span class="validation-name">{html.escape(subcategory["name"])}</span>'
                + _values(subcategory, subcategory_maximum)
                + '</summary><div class="validation-children">'
                + "".join(category_sections)
                + "</div></details>"
            )
        family_sections.append(
            '<details class="validation-node family-node">'
            '<summary class="validation-row">'
            f'<span class="validation-name">{html.escape(family["name"])}</span>'
            + _values(family, family_maximum)
            + '</summary><div class="validation-children">'
            + "".join(subcategory_sections)
            + "</div></details>"
        )
    return "".join(family_sections)


def _style() -> str:
    return """
<style>
:root{color-scheme:light dark;--muted:#536965;--surface:#edf6f4;--surface-2:#e4f0ed;--track:#d2e1de;--target:#7b8d89;--actual:#218f84;--warning:#bd711f}
@media(prefers-color-scheme:dark){:root{--muted:#a7bbb7;--surface:#142420;--surface-2:#1a2d29;--track:#2a403c;--target:#91a29f;--actual:#5cc8bb;--warning:#e5a456}}
*{box-sizing:border-box}body{max-width:1480px;margin:0 auto;padding:34px 26px 72px;background:Canvas;color:CanvasText;font:14px/1.42 system-ui,sans-serif}h1{margin:0 0 20px;font-size:28px;line-height:1.2}.aggregate-row{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:12px 28px;margin-bottom:16px}.plot-legend{display:flex;gap:18px;margin:0 0 8px;color:var(--muted)}.plot-legend span::before{display:inline-block;width:18px;height:5px;margin-right:7px;border-radius:999px;content:"";vertical-align:middle}.target-key::before{background:var(--target)}.actual-key::before{background:var(--actual)}.provenance-note{max-width:940px;margin:0 0 18px;color:var(--muted)}.validation-hierarchy{display:grid;gap:7px}.validation-node{border:0}.validation-row{display:grid;grid-template-columns:minmax(250px,1.35fr) repeat(5,minmax(128px,1fr));gap:8px 22px;align-items:center;min-width:0;padding:12px 14px;list-style-position:inside}.validation-node>summary{cursor:pointer}.family-node>summary{border-radius:9px;background:var(--surface)}.family-node>summary:hover,.subcategory-node>summary:hover,.category-node>summary:hover{background:var(--surface-2)}.subcategory-node>summary{margin-top:6px;border-radius:8px;background:color-mix(in srgb,var(--surface) 62%,transparent)}.category-node>summary{border-radius:7px}.validation-name{min-width:0;overflow-wrap:anywhere;font-weight:600}.validation-context{display:block;margin:2px 0 0 18px;color:var(--muted);font-size:12px;font-weight:400}.validation-metric{min-width:0;font-variant-numeric:tabular-nums}.validation-label,.validation-value{display:block}.validation-label{color:var(--muted);font-size:12px}.validation-value{margin-top:2px}.validation-warning .validation-value{color:var(--warning)}.target-actual-bars{grid-column:2/-1;display:grid;gap:3px}.target-track,.actual-track{display:block;height:5px;border-radius:999px;background:var(--track);overflow:hidden}.target-fill,.actual-fill{display:block;height:100%;border-radius:inherit}.target-fill{background:var(--target)}.actual-fill{background:var(--actual)}.validation-children{margin-left:22px}.lower-groups{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:3px 22px}.lower-group-row{grid-template-columns:minmax(190px,1.15fr) repeat(5,minmax(105px,1fr));padding-top:9px;padding-bottom:9px}.lower-group-row .target-actual-bars{grid-column:2/-1}
@media(max-width:1050px){.validation-row,.lower-group-row{grid-template-columns:minmax(220px,1fr) repeat(2,minmax(130px,1fr))}.validation-row>.validation-metric:nth-of-type(n+4){margin-top:5px}.target-actual-bars,.lower-group-row .target-actual-bars{grid-column:2/-1}.lower-groups{grid-template-columns:1fr}}
@media(max-width:700px){body{padding:24px 14px 48px}.aggregate-row{grid-template-columns:repeat(2,minmax(0,1fr))}.validation-row,.lower-group-row{grid-template-columns:1fr 1fr;gap:8px 14px}.validation-name{grid-column:1/-1}.target-actual-bars,.lower-group-row .target-actual-bars{grid-column:1/-1}.validation-children{margin-left:8px}}
</style>
"""


def render_output_validation_report(
    inventory_details: dict[str, Any],
    validation_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Render exact output totals against the inventory and sampling plan."""

    families, flat_rows = _build_hierarchy(inventory_details, validation_rows)
    total_source = sum(row["source"] for row in families)
    total_target = sum(row["target"] for row in families)
    total_actual = sum(row["actual"] for row in families)
    report = (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Dolma 3.5 source, target, and materialized output</title>" + _style() + "</head><body>"
        "<h1>Dolma 3.5 Source, Target, and Materialized Output</h1>"
        '<div class="aggregate-row">'
        + _metric("Source", f"{_human_count(total_source)} tokens")
        + _metric("Target", f"{_human_count(total_target)} tokens")
        + _metric("Actual", f"{_human_count(total_actual)} tokens")
        + _metric(
            "Sampling",
            _sampling_label(total_source, total_target, total_actual),
        )
        + _metric("Actual − target", _delta_label(total_target, total_actual))
        + "</div>"
        '<div class="plot-legend"><span class="target-key">Target</span>'
        '<span class="actual-key">Actual</span></div>'
        '<p class="provenance-note">Actual output is exact through the sampling tier. '
        "Lower-group rows retain the source and target plan because merged output does not retain "
        "per-input provenance.</p>"
        '<main class="validation-hierarchy">' + _render_hierarchy(families) + "</main></body></html>\n"
    )
    return flat_rows, report
