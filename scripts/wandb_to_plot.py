import argparse
import bisect
import fnmatch
from collections import defaultdict
from functools import partial
from pathlib import Path
from statistics import stdev
from typing import Dict, Iterable, List, Optional, Tuple

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
    ap.add_argument("--legend-text", type=str, default="Runs", help="Legend text")
    ap.add_argument("--y-log-scale", action="store_true", help="Use log scale for y axis")
    ap.add_argument("-v", "--vocabulary", type=Path, default=None, help="Path json file with pretty names")
    return ap.parse_args()


def match_run_name(name: str, run_names: List[str]) -> Optional[str]:
    for run_name in run_names:
        if fnmatch.filter([name], run_name):
            return run_name
    return None


def remove_outliers(
    x: Iterable[float], y: Iterable[float], z: float = 10.0
) -> Tuple[Iterable[float], Iterable[float]]:
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
    runs = api.runs(
        path=f"{opts.wandb_team}/{opts.wandb_project}",
        filters={"$or": [{"config.run_name": {"$regex": n}} for n in opts.wandb_names]},
    )

    vocabulary: Dict[str, str] = {}
    if opts.vocabulary is not None and opts.vocabulary.exists():
        with opts.vocabulary.open("r") as f:
            vocabulary = yaml.safe_load(f.read())

    metrics = defaultdict(lambda: {n: {"x": [], "y": []} for n in opts.wandb_names})
    run_name_matcher = partial(match_run_name, run_names=opts.wandb_names)

    for wb_run in runs:
        plot_group_name = run_name_matcher(wb_run.config["run_name"])
        print(f"Processing run {wb_run.config['run_name']} into group {plot_group_name}")

        if opts.samples > 0:
            x_axis_history = wb_run.history(samples=opts.samples, keys=[opts.x_axis], pandas=False)
        else:
            x_axis_history = list(wb_run.scan_history(keys=[opts.x_axis]))
        steps, x_axis = zip(*sorted([(wb_step["_step"], wb_step[opts.x_axis]) for wb_step in x_axis_history]))

        if opts.samples > 0:
            history = wb_run.history(samples=opts.samples, keys=opts.y_axis, pandas=False)
        else:
            history = wb_run.scan_history(keys=opts.y_axis)

        for y_axis in opts.y_axis:
            yaxis_pretty_name = vocabulary.get(y_axis, y_axis)
            for wb_step in history:
                loc = min(bisect.bisect_left(steps, wb_step["_step"]), len(x_axis) - 1)
                metrics[yaxis_pretty_name][plot_group_name]["x"].append(x_axis[loc])
                metrics[yaxis_pretty_name][plot_group_name]["y"].append(wb_step[y_axis])

    opts.destination.mkdir(parents=True, exist_ok=True)
    xaxis_pretty_name = vocabulary.get(opts.x_axis, opts.x_axis)

    for y_axis, plot_groups in metrics.items():
        fig = go.Figure()

        # these we figure out as we go
        use_y_log = opts.y_log_scale
        top_right_legend = False

        for run_name, run_data in plot_groups.items():
            # start by sorting the data by x axis
            x, y = zip(*sorted(zip(run_data["x"], run_data["y"])))

            if opts.remove_outliers:
                x, y = remove_outliers(x=x, y=y)

            if opts.max_x_axis:
                x, y = zip(*[(x, y) for x, y in zip(x, y) if x <= opts.max_x_axis])

            if opts.max_y_axis:
                x, y = zip(*[(x, y) for x, y in zip(x, y) if y <= opts.max_y_axis])

            if y[-1] < y[0]:
                top_right_legend = True

            use_y_log = use_y_log or (max(y) / min(y) > 100)
            figure_run_name = vocabulary.get(run_name, run_name)
            fig.add_trace(go.Scatter(name=figure_run_name, x=x, y=y))

        if top_right_legend:
            fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        else:
            fig.update_layout(legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))

        fig.update_layout(xaxis_title=xaxis_pretty_name, yaxis_title=y_axis, legend_title_text=opts.legend_text)
        fig.update_yaxes(type="log" if use_y_log else "linear")
        fig.update_xaxes(range=([0, opts.max_x_axis] if opts.max_x_axis is not None else None))

        dest = opts.destination / f"{y_axis.replace('/', '_')}.{opts.plot_extension}"
        fig.write_image(str(dest))
        print(f"Saved {y_axis} plot to {dest}")


if __name__ == "__main__":
    main()
