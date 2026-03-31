#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "This script requires matplotlib in the Python environment used for plotting."
    ) from exc


SUMMARY_COLUMNS = {
    "model",
    "batch_size",
    "runtime_s",
    "steps_completed",
    "forward_time_mean_ns",
    "forward_time_stdev_ns",
    "backward_time_mean_ns",
    "backward_time_stdev_ns",
    "optimizer_time_mean_ns",
    "optimizer_time_stdev_ns",
}

STEPS_COLUMNS = {
    "step",
    "step_end_time_s",
    "step_time_ns",
    "forward_time_ns",
    "backward_time_ns",
    "optimizer_time_ns",
}

TIMELINE_COLUMNS = {
    "time_since_start_s",
    "step",
    "state",
    "process_cpu_util_pct",
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "gpu_mem_util_pct",
    "gpu_energy_mj_from_start",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots for one profile run from the raw profile CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing run_<N>_profile_summary/steps/timeline.csv files.",
    )
    parser.add_argument(
        "--run-num",
        type=int,
        required=True,
        help="Run number used in the profile CSV filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the generated plots will be written. Defaults to <input-dir>/plots/run_<N>.",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame, required_columns: set[str], frame_name: str) -> None:
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def load_run_data(input_dir: Path, run_num: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = input_dir / f"run_{run_num}_profile_summary.csv"
    steps_path = input_dir / f"run_{run_num}_profile_steps.csv"
    timeline_path = input_dir / f"run_{run_num}_profile_timeline.csv"

    for path in (summary_path, steps_path, timeline_path):
        if not path.exists():
            raise FileNotFoundError(f"Expected profile output file not found: {path}")

    summary = pd.read_csv(summary_path)
    steps = pd.read_csv(steps_path)
    timeline = pd.read_csv(timeline_path)

    validate_columns(summary, SUMMARY_COLUMNS, "summary CSV")
    validate_columns(steps, STEPS_COLUMNS, "steps CSV")
    validate_columns(timeline, TIMELINE_COLUMNS, "timeline CSV")

    if len(summary) != 1:
        raise ValueError(f"summary CSV should contain exactly one row, found {len(summary)}")

    return summary, steps, timeline


def validate_run_consistency(summary: pd.DataFrame, steps: pd.DataFrame, timeline: pd.DataFrame) -> None:
    expected_steps = int(summary.loc[0, "steps_completed"])
    if len(steps) != expected_steps:
        raise ValueError(
            f"steps CSV row count ({len(steps)}) does not match steps_completed ({expected_steps})"
        )

    if int(steps["step"].iloc[-1]) != expected_steps:
        raise ValueError(
            f"last step id ({int(steps['step'].iloc[-1])}) does not match steps_completed ({expected_steps})"
        )

    if timeline.empty:
        raise ValueError("timeline CSV is empty")


def plot_timelines(
    summary: pd.DataFrame,
    timeline: pd.DataFrame,
    output_dir: Path,
    run_num: int,
) -> None:
    timeline = timeline.sort_values("time_since_start_s").reset_index(drop=True)

    model_name = summary.loc[0, "model"]
    batch_size = int(summary.loc[0, "batch_size"])
    runtime_s = float(summary.loc[0, "runtime_s"])

    time_s = timeline["time_since_start_s"]

    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)

    axes[0].plot(time_s, timeline["process_cpu_util_pct"], color="tab:blue")
    axes[0].set_ylabel("CPU util (%)")
    axes[0].set_title("Process CPU utilization")

    axes[1].plot(time_s, timeline["gpu_util_pct"], color="tab:green")
    axes[1].set_ylabel("GPU util (%)")
    axes[1].set_title("GPU utilization")

    axes[2].plot(time_s, timeline["gpu_mem_used_mb"], color="tab:orange")
    axes[2].set_ylabel("GPU mem (MB)")
    axes[2].set_title("GPU memory used")

    axes[3].plot(time_s, timeline["gpu_energy_mj_from_start"] / 1_000_000, color="tab:red")
    axes[3].set_ylabel("Energy (kJ)")
    axes[3].set_xlabel("Time since start (s)")
    axes[3].set_title("GPU energy from start")

    figure.suptitle(
        f"{model_name} profile run {run_num} (batch size {batch_size}, runtime {runtime_s:.2f} s)"
    )
    figure.savefig(output_dir / f"run_{run_num}_profile_timelines.png", dpi=200)
    plt.close(figure)


def plot_phase_bars(summary: pd.DataFrame, output_dir: Path, run_num: int) -> None:
    model_name = summary.loc[0, "model"]
    batch_size = int(summary.loc[0, "batch_size"])

    phases = ["forward", "backward", "optimizer"]
    means_ms = [
        float(summary.loc[0, "forward_time_mean_ns"]) / 1_000_000,
        float(summary.loc[0, "backward_time_mean_ns"]) / 1_000_000,
        float(summary.loc[0, "optimizer_time_mean_ns"]) / 1_000_000,
    ]
    stdevs_ms = [
        float(summary.loc[0, "forward_time_stdev_ns"]) / 1_000_000,
        float(summary.loc[0, "backward_time_stdev_ns"]) / 1_000_000,
        float(summary.loc[0, "optimizer_time_stdev_ns"]) / 1_000_000,
    ]

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(phases, means_ms, yerr=stdevs_ms, color=["tab:blue", "tab:green", "tab:red"], capsize=6)
    axis.set_ylabel("Phase time (ms)")
    axis.set_title(f"{model_name} phase times for run {run_num} (batch size {batch_size})")
    figure.savefig(output_dir / f"run_{run_num}_profile_phase_bars.png", dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "plots" / f"run_{args.run_num}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, steps, timeline = load_run_data(input_dir, args.run_num)
    validate_run_consistency(summary, steps, timeline)

    plot_timelines(summary, timeline, output_dir, args.run_num)
    plot_phase_bars(summary, output_dir, args.run_num)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
