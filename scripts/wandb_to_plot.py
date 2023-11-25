import argparse
import re
from necessary import necessary

with necessary(["plotly", "wandb"]):
    import plotly
    import wandb


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-t", "--wandb-team", type=str, required=True, help="Name of the wandb team to use, e.g. 'ai2-llm'"
    )
    ap.add_argument(
        "-p",
        "--wandb-project",
        type=str,
        required=True,
        help="Name of the wandb project to use, e.g. 'olmo-small'",
    )
    ap.add_argument(
        "-n", "--wandb-name", type=str, required=True, help="Run name or regex to use, e.g. '3T-lower-tie-.*'"
    )
    ap.add_argument(
        "-x",
        "--x-axis",
        type=str,
        default="throughput/total_tokens",
    )
    ap.add_argument(
        "-y",
        "--y-axis",
        nargs="+",
        type=str,
        default=["train/Perplexity"]
    )
    return ap.parse_args()


def main():
    opts = parse_args()

    # make sure we're logged in
    wandb.login()

    api = wandb.Api()
    runs = api.runs(f"{opts.wandb_team}/{opts.wandb_project}")

    re_run_name = re.compile(opts.wandb_name)

    for wb_run in runs:
        if not re_run_name.search(wb_run.name):
            continue

        import ipdb

        ipdb.set_trace()


if __name__ == "__main__":
    main()
