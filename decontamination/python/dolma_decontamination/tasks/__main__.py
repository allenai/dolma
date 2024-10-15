from .base import Task
from .registry import task_registry
import re
import argparse
from rich.console import Console
from rich.table import Table
from rich.text import Text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str, choices=["list", "run"])
    parser.add_argument("-t", "--task", type=str, nargs="*")
    parser.add_argument("-n", "--num-to-preview", type=int, default=10)
    args = parser.parse_args()
    args.task = [re.compile(t) for t in args.task] if args.task else None
    return args


def print_tasks(tasks_names: list[str], max_datasets_to_show: int = 3, max_targets_to_show: int = 3):
    tasks = [(task_name, task_registry.get_task(task_name)()) for task_name in tasks_names]

    console = Console()
    table = Table(title="Tasks")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Datasets", style="yellow")
    table.add_column("Targets", style="magenta")

    max_width = console.width
    task_name_width = min(max(len(task_name) for task_name, _ in tasks), int(max_width * 0.3))
    available_width = max_width - task_name_width - 6  # 8 for padding and borders
    dw = available_width // 2
    tw = available_width - dw

    for task_name, task in tasks:
        datasets_str = Text()
        for i, dataset in enumerate(task.datasets):
            if i < max_datasets_to_show:
                datasets_str += Text(f"{d[:dw - 1] + '…' if len(d := str(dataset)) > dw else d}")
            else:
                datasets_str += Text(
                    f"({len(task.datasets) - max_datasets_to_show:,} more datasets not shown)",
                    style='italic'
                )
                break
            if i < max_datasets_to_show:
                datasets_str += Text("\n")

        targets_str = Text()
        max_targets_to_show_for_this_task = min(max_targets_to_show, len(task.datasets) + 1)
        for i, target in enumerate(task.targets):
            if i < max_targets_to_show_for_this_task:
                targets_str += Text(
                    f"{t[:tw - 1] + '…' if len(t := str(target).replace('\n', '\\n')) > tw else t}"
                )
            else:
                targets_str += Text(
                    f"({len(task.targets) - max_targets_to_show_for_this_task:,} more targets not shown)",
                    style='italic'
                )
                break
            if i < max_targets_to_show_for_this_task:
                targets_str += Text("\n")

        table.add_row(Text(task_name, style='bold'), datasets_str, targets_str)

    console.print(table)


def print_samples(tasks_names: list[str], num_to_preview: int):
    tasks = [task_registry.get_task(task_name)() for task_name in tasks_names]

    console = Console()
    for task in tasks:
        table = Table(title=task.name, show_lines=True)
        table.add_column(Text("Dataset ID", style="bold"), style="green")
        table.add_column(Text("Target Label", style="bold"), style="yellow")
        table.add_column(Text("Text", style="bold"), style="magenta")

        for i, doc in enumerate(task.docs()):
            if i >= num_to_preview:
                break
            table.add_row(str(doc.meta.dataset_id), str(doc.meta.target_label), doc.text)

        console.print(table)


def main():
    args = parse_args()
    tasks_names = [
        t for t in task_registry.list_tasks()
        if (args.task is None or any(r.match(t) for r in args.task))
    ]
    if args.action == "list":
        return print_tasks(tasks_names)

    if args.action == "run":
        return print_samples(tasks_names, args.num_to_preview)


if __name__ == "__main__":
    main()
