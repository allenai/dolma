import argparse
import bisect
import fnmatch
import re
from collections import defaultdict
from functools import lru_cache, partial
from math import ceil, floor, log10
from pathlib import Path
from statistics import stdev
from typing import List, Optional, Sequence, Tuple

from necessary import necessary

with necessary(
    modules=["plotly", "wandb", "kaleido", "yaml"],
    message="Please run `pip install plotly wandb kaleido pyyaml`",
):
    import plotly.graph_objs as go
    import plotly.io as pio
    import wandb
    import yaml

    pio.kaleido.scope.mathjax = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--wandb-team", type=str, required=True, help="Name of the W&B team")
    ap.add_argument("-p", "--wandb-project", type=str, required=True, help="Name of the W&B project")
    ap.add_argument("-n", "--wandb-names", type=str, nargs="+", required=True, help="Run name or regex")
    ap.add_argument("-x", "--x-axis", type=str, default="throughput/total_tokens", help="X axis")
    ap.add_argument("-y", "--y-axis", nargs="+", type=str, default=["train/Perplexity"], help="Y axis")
    ap.add_argument("-s", "--samples", default=0, type=int, help="Samples to get; 0 for no sampling")
    ap.add_argument("-d", "--destination", type=Path, required=True, help="Directory to save plots to")
    ap.add_argument("-e", "--plot-extension", type=str, default="pdf", help="Plot extension to use")
    ap.add_argument("--max-x-axis", type=float, default=None, help="Maximum x value")
    ap.add_argument("--max-y-axis", type=float, default=None, help="Maximum y value")
    ap.add_argument("--remove-outliers", action="store_true", help="Remove outliers")
    ap.add_argument("--legend-title", type=str, default=None, help="Legend title")
    ap.add_argument("--y-log-scale", action="store_true", help="Use log scale for y axis")
    ap.add_argument("-v", "--vocabulary", type=Path, default=None, help="Path json file with pretty names")
    ap.add_argument("-N", "--experiment-nickname", type=str, default=None, help="Experiment nickname")
    ap.add_argument("--plotly-theme", type=str, default="none", help="Plotly theme to use")
    ap.add_argument("--plotly-font-size", type=int, default=10, help="Plotly font size")
    ap.add_argument("--plotly-figure-width", type=int, default=800, help="Plotly figure width")
    ap.add_argument("--plotly-figure-height", type=int, default=500, help="Plotly figure height")
    return ap.parse_args()


@lru_cache(maxsize=1)
def translate_to_regex(s: str) -> re.Pattern:
    return re.compile(fnmatch.translate(s))


def match_run_name(name: str, run_names: List[str]) -> Optional[str]:
    for run_name in run_names:
        if translate_to_regex(run_name).search(name):
            return run_name
    return None


def remove_outliers(
    x: Sequence[float], y: Sequence[float], z: float = 10.0
) -> Tuple[Sequence[float], Sequence[float]]:
    if min(y) >= 0 and max(y) <= 1:
        # do not crop values that are in [0, 1]
        return x, y

    std = stdev(y)
    x, y = zip(*[(x, y) for x, y in zip(x, y) if abs(y - std) < z * std])
    return x, y


def main():
    opts = parse_args()

    # make sure we're logged in
    wandb.login()

    # get all the runs matching name filters
    api = wandb.Api()
    wb_path = f"{opts.wandb_team}/{opts.wandb_project}"
    # wb_filters = {"$or": [{"config.run_name": {"$regex": n}} for n in opts.wandb_names]}
    wb_filters = {"$or": [{"display_name": {"$regex": n}} for n in opts.wandb_names]}
    wb_runs = api.runs(path=wb_path, filters=wb_filters)

    vocabulary = {}
    if opts.vocabulary is not None and opts.vocabulary.exists():
        with opts.vocabulary.open("r") as f:
            vocabulary = yaml.safe_load(f.read())

    # prepare the vocabulary by (a) overriding with experiment specific
    # names that are not in the vocabulary file and (b) experiment config
    # (nested vocabulary)
    vocabulary = {
        **{k: v for k, v in vocabulary.items() if isinstance(v, str)},
        **{k: v for k, v in vocabulary.get(opts.experiment_nickname, {}).items() if isinstance(v, str)},
    }

    metrics_values = defaultdict(lambda: {n: {"x": [], "y": []} for n in opts.wandb_names})
    metrics_names = defaultdict(lambda: {n: "" for n in opts.wandb_names})
    run_name_matcher = partial(match_run_name, run_names=opts.wandb_names)

    print(f"Found {len(wb_runs)} matching runs in {wb_path}")

    for wb_run in wb_runs:
        plot_group_name = run_name_matcher(wb_run.name)

        if plot_group_name is None:
            print(f"WARNING: could not find a name match for {wb_run.name}")
            continue
        print(f"Processing run {wb_run.name} into group {plot_group_name}")

        if opts.samples > 0:
            x_axis_history = wb_run.history(samples=opts.samples, keys=[opts.x_axis], pandas=False)
        else:
            x_axis_history = list(wb_run.scan_history(keys=[opts.x_axis]))

        if len(x_axis_history) == 0:
            # this run has crashed and has no history
            print(f"WARNING: skipping {wb_run.name} because it has no history. Crashed early?")
            continue

        steps, x_axis = zip(*sorted([(wb_step["_step"], wb_step[opts.x_axis]) for wb_step in x_axis_history]))

        if opts.samples > 0:
            history = wb_run.history(samples=opts.samples, keys=opts.y_axis, pandas=False)
        else:
            history = wb_run.scan_history(keys=opts.y_axis)

        for y_axis in opts.y_axis:
            yaxis_pretty_name = vocabulary.get(y_axis, y_axis)

            inferred_metric_name = ""
            if "perplexity" in y_axis.lower():
                inferred_metric_name = "Perplexity"
            elif "_f1" in y_axis.lower():
                inferred_metric_name = "F1 Score"
            elif "crossentropyloss" in y_axis.lower():
                inferred_metric_name = "Cross Entropy"
            elif "downstream" in y_axis.lower():
                inferred_metric_name = "Accuracy"

            metric_name = vocabulary.get("metrics", {}).get(y_axis, inferred_metric_name)
            metrics_names[yaxis_pretty_name][plot_group_name] = metric_name

            for wb_step in history:
                loc = min(bisect.bisect_left(steps, wb_step["_step"]), len(x_axis) - 1)
                metrics_values[yaxis_pretty_name][plot_group_name]["x"].append(x_axis[loc])
                metrics_values[yaxis_pretty_name][plot_group_name]["y"].append(wb_step[y_axis])

    xaxis_pretty_name = vocabulary.get(opts.x_axis, opts.x_axis)

    for y_axis, plot_groups in metrics_values.items():
        fig = go.Figure()

        # these we figure out as we go
        use_y_log = opts.y_log_scale
        top_right_legend = False
        metric_name = None
        global_min_y = float("inf")
        global_max_y = float("-inf")

        for run_name, run_data in plot_groups.items():
            if metric_name is not None:
                assert metrics_names[y_axis][run_name] == metric_name, "Inconsistent metric names"
            metric_name = metrics_names[y_axis][run_name]

            if len(run_data["y"]) == 0:
                print(f"WARNING: skipping {run_name} because it has no data for {y_axis}")
                continue

            # start by sorting the data by x axis
            x, y = zip(*sorted(zip(run_data["x"], run_data["y"])))

            if opts.remove_outliers:
                x, y = remove_outliers(x=x, y=y)

            if opts.max_x_axis and min(x) < opts.max_x_axis:
                x, y = zip(*[(x, y) for x, y in zip(x, y) if x <= opts.max_x_axis])

            if opts.max_y_axis and min(y) < opts.max_y_axis:
                x, y = zip(*[(x, y) for x, y in zip(x, y) if y <= opts.max_y_axis])

            if y[-1] < y[0]:
                # decreasing y values, so we want the legend on the top right
                top_right_legend = True

            # only use log scale if we have values that are not in [0, 1]
            if max(y) > 1 or min(y) < 0:
                min_y = min([y for y in y if y > 0] or [1e-3])  # avoid diving by zero
                use_y_log = use_y_log or (max(y) / min_y > 100)

            # keep track of global min and max
            global_min_y = min(global_min_y, min(y))
            global_max_y = max(global_max_y, max(y))

            figure_run_name = vocabulary.get(run_name, run_name)
            fig.add_trace(go.Scatter(name=figure_run_name, x=x, y=y, mode="lines"))

        legend_config = {
            "yanchor": "top" if top_right_legend else "bottom",
            "y": 0.99 if top_right_legend else 0.01,
            "xanchor": "right",
            "x": 0.99,
            "font": {"size": opts.plotly_font_size},
            "borderwidth": 1,
            "bordercolor": "Gray",
        }
        fig.update_layout(legend=legend_config)

        fig.update_layout(
            template=opts.plotly_theme,
            xaxis_title=xaxis_pretty_name,
            yaxis_title=metric_name,
            legend_title=opts.legend_title,
            title_text=y_axis.split('@@@')[0],
            font=dict(size=opts.plotly_font_size),
            width=opts.plotly_figure_width,
            height=opts.plotly_figure_height,
            margin=dict(
                l=4 * opts.plotly_font_size,
                r=opts.plotly_font_size,
                b=4 * opts.plotly_font_size,
                t=3 * opts.plotly_font_size,
            ),
        )
        if use_y_log:
            steps = []
            for decade in range(ceil(global_max_y / global_min_y if global_min_y > 0 else log10(global_max_y))):
                unit = 10**decade
                start = max(unit, floor(global_min_y / unit) * unit)
                end = min(10 ** (decade + 1), ceil(global_max_y / unit) * unit)
                steps.extend(range(int(start), int(end) + unit, unit))

            fig.update_yaxes(type="log")
            fig.update_layout(yaxis={"tickmode": "array", "tickvals": steps})

        fig.update_xaxes(range=([0, opts.max_x_axis] if opts.max_x_axis is not None else None))

        file_name = re.sub(r"\W+", "_", y_axis).lower().strip("_")
        file_path = opts.destination / f"{file_name}.{opts.plot_extension}"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(file_path))
        print(f"Saved {y_axis} plot to {file_path}")


NORMALIZE_PLOTLY_NAME = re.compile(r"([a-z])([A-Z])")

if __name__ == "__main__":
    main()
