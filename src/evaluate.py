import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf

matplotlib.use("Agg")


def load_wandb_config() -> Dict[str, str]:
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    cfg = OmegaConf.load(config_path)
    return {"entity": cfg.wandb.entity, "project": cfg.wandb.project, "mode": cfg.wandb.mode}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def parse_kv_args(argv: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key.lstrip("-")] = value
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--run_ids", type=str, default=None)
    args, unknown = parser.parse_known_args()
    if args.results_dir is None or args.run_ids is None:
        fallback = parse_kv_args(unknown)
        if args.results_dir is None:
            args.results_dir = fallback.get("results_dir")
        if args.run_ids is None:
            args.run_ids = fallback.get("run_ids")
    if args.results_dir is None or args.run_ids is None:
        raise ValueError("results_dir and run_ids must be provided as key=value or --results_dir/--run_ids.")
    return args


def plot_time_series(history: pd.DataFrame, keys: List[str], title: str, filename: Path) -> bool:
    if history.empty:
        return False
    step_col = "_step" if "_step" in history.columns else None
    plt.figure(figsize=(8, 4))
    plotted = False
    for key in keys:
        if key in history.columns:
            x = history[step_col] if step_col else range(len(history))
            y = history[key]
            plt.plot(x, y, label=key)
            if len(y) > 0:
                plt.annotate(
                    f"{y.iloc[-1]:.3f}",
                    xy=(x.iloc[-1] if step_col else len(y) - 1, y.iloc[-1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
            plotted = True
    if not plotted:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return True


def plot_confusion_matrix(counts: Dict[str, Any], filename: Path) -> bool:
    required = [
        "draft_correct_final_correct",
        "draft_correct_final_incorrect",
        "draft_incorrect_final_correct",
        "draft_incorrect_final_incorrect",
    ]
    if not all(k in counts for k in required):
        return False
    matrix = np.array(
        [
            [counts["draft_correct_final_correct"], counts["draft_correct_final_incorrect"]],
            [counts["draft_incorrect_final_correct"], counts["draft_incorrect_final_incorrect"]],
        ]
    )
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Final correct / incorrect")
    plt.ylabel("Draft correct / incorrect")
    plt.title("Draft vs Final Correctness")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return True


def plot_bar(values: Dict[str, float], title: str, filename: Path, ylabel: str) -> bool:
    if not values:
        return False
    plt.figure(figsize=(8, 4))
    labels = list(values.keys())
    vals = [values[k] for k in labels]
    sns.barplot(x=labels, y=vals)
    for idx, val in enumerate(vals):
        plt.text(idx, val, f"{val:.3f}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return True


def plot_metric_table(metrics: Dict[str, Dict[str, float]], filename: Path) -> bool:
    if not metrics:
        return False
    df = pd.DataFrame(metrics)
    if df.empty:
        return False
    plt.figure(figsize=(10, 0.6 + 0.25 * len(df)))
    plt.axis("off")
    table = plt.table(
        cellText=np.round(df.values, 4),
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return True


def plot_boxplot(df: pd.DataFrame, title: str, filename: Path, ylabel: str) -> bool:
    if df.empty:
        return False
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="group", y="value", data=df)
    sns.stripplot(x="group", y="value", data=df, color="black", size=4, jitter=0.2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return True


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def two_proportion_z_test(count1: int, n1: int, count2: int, n2: int) -> Dict[str, float]:
    if n1 == 0 or n2 == 0:
        return {"z": 0.0, "p": 1.0}
    p1 = count1 / n1
    p2 = count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / denom if denom > 0 else 0.0
    p = 2 * (1 - norm_cdf(abs(z)))
    return {"z": z, "p": p}


def fetch_run(api: wandb.Api, entity: str, project: str, run_id: str) -> tuple[wandb.apis.public.Run, Dict[str, Any]]:
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as exc:
        raise RuntimeError(
            f"Run {run_id} not found in WandB. Evaluation requires full runs with WandB enabled; "
            "trial mode disables logging."
        ) from exc
    config = dict(run.config)
    wandb_cfg = config.get("wandb", {})
    if config.get("mode") == "trial" or (isinstance(wandb_cfg, dict) and wandb_cfg.get("mode") == "disabled"):
        raise RuntimeError(
            f"Run {run_id} was executed in trial mode or with WandB disabled. "
            "Evaluation requires full runs with logging enabled."
        )
    return run, config


def fetch_full_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    history = run.history()
    if history is None:
        return pd.DataFrame()
    if "_step" not in history.columns:
        history["_step"] = range(len(history))
    return history


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)

    wandb_cfg = load_wandb_config()
    api = wandb.Api()

    all_metrics: Dict[str, Dict[str, float]] = {}
    summary_records: Dict[str, Dict[str, Any]] = {}
    generated_paths: List[Path] = []

    for run_id in run_ids:
        run, config = fetch_run(api, wandb_cfg["entity"], wandb_cfg["project"], run_id)
        history = fetch_full_history(run)
        summary = run.summary._json_dict

        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        history_records = history.to_dict(orient="records") if not history.empty else []
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, {"run_id": run_id, "summary": summary, "config": config, "history": history_records})
        generated_paths.append(metrics_path)

        if plot_time_series(
            history,
            ["accuracy", "draft_accuracy"],
            "Accuracy over Steps",
            run_dir / f"{run_id}_learning_curve.pdf",
        ):
            generated_paths.append(run_dir / f"{run_id}_learning_curve.pdf")

        if plot_time_series(
            history,
            ["certified_coverage", "agreement_rate_when_both_valid"],
            "Certification Diagnostics",
            run_dir / f"{run_id}_certification_curve.pdf",
        ):
            generated_paths.append(run_dir / f"{run_id}_certification_curve.pdf")

        if plot_confusion_matrix(summary, run_dir / f"{run_id}_confusion_matrix.pdf"):
            generated_paths.append(run_dir / f"{run_id}_confusion_matrix.pdf")

        final_metrics = {}
        for key in [
            "accuracy",
            "draft_accuracy",
            "certified_coverage",
            "agreement_rate_when_both_valid",
            "exec_rate_a",
            "exec_rate_b",
            "harm_rate",
        ]:
            if key in summary and isinstance(summary[key], (int, float)):
                final_metrics[key] = float(summary[key])
        if final_metrics and plot_bar(
            final_metrics,
            "Final Metrics",
            run_dir / f"{run_id}_final_metrics_bar.pdf",
            "value",
        ):
            generated_paths.append(run_dir / f"{run_id}_final_metrics_bar.pdf")

        numeric_summary = {
            k: float(v)
            for k, v in summary.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        summary_records[run_id] = numeric_summary
        for key, value in numeric_summary.items():
            all_metrics.setdefault(key, {})[run_id] = value

    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    primary_metric = "accuracy"
    proposed_runs = {rid: summary_records[rid] for rid in run_ids if "proposed" in rid.lower()}
    baseline_runs = {
        rid: summary_records[rid]
        for rid in run_ids
        if ("baseline" in rid.lower() or "comparative" in rid.lower())
    }

    best_proposed = {"run_id": None, "value": None}
    if proposed_runs:
        best_run = max(proposed_runs.items(), key=lambda x: x[1].get(primary_metric, float("-inf")))
        best_proposed = {"run_id": best_run[0], "value": best_run[1].get(primary_metric)}

    best_baseline = {"run_id": None, "value": None}
    if baseline_runs:
        best_run = max(baseline_runs.items(), key=lambda x: x[1].get(primary_metric, float("-inf")))
        best_baseline = {"run_id": best_run[0], "value": best_run[1].get(primary_metric)}

    gap = None
    minimize_metrics = {"loss", "error", "perplexity"}
    if best_proposed["value"] is not None and best_baseline["value"] not in (None, 0):
        raw_gap = (best_proposed["value"] - best_baseline["value"]) / best_baseline["value"] * 100
        if primary_metric.lower() in minimize_metrics:
            raw_gap = -raw_gap
        gap = raw_gap

    significance_tests = []
    if proposed_runs and baseline_runs:
        for prop_id, prop_summary in proposed_runs.items():
            for base_id, base_summary in baseline_runs.items():
                count1 = int(prop_summary.get("correct_final", 0))
                n1 = int(prop_summary.get("total_examples", 0))
                count2 = int(base_summary.get("correct_final", 0))
                n2 = int(base_summary.get("total_examples", 0))
                stats = two_proportion_z_test(count1, n1, count2, n2)
                significance_tests.append(
                    {
                        "proposed": prop_id,
                        "baseline": base_id,
                        "z": stats["z"],
                        "p": stats["p"],
                        "diff": (count1 / n1 - count2 / n2) if n1 and n2 else 0.0,
                    }
                )

    aggregated_metrics = {
        "primary_metric": primary_metric,
        "metrics": all_metrics,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
        "significance_tests": significance_tests,
    }

    aggregated_path = comparison_dir / "aggregated_metrics.json"
    write_json(aggregated_path, aggregated_metrics)
    generated_paths.append(aggregated_path)

    if primary_metric in all_metrics and plot_bar(
        all_metrics[primary_metric],
        "Primary Metric Comparison",
        comparison_dir / "comparison_accuracy_bar_chart.pdf",
        primary_metric,
    ):
        generated_paths.append(comparison_dir / "comparison_accuracy_bar_chart.pdf")

    if all_metrics and plot_metric_table(all_metrics, comparison_dir / "comparison_metrics_table.pdf"):
        generated_paths.append(comparison_dir / "comparison_metrics_table.pdf")

    if summary_records:
        rows = []
        for run_id, metrics in summary_records.items():
            value = metrics.get(primary_metric)
            if value is None:
                continue
            if "proposed" in run_id.lower():
                group = "proposed"
            elif "baseline" in run_id.lower() or "comparative" in run_id.lower():
                group = "baseline"
            else:
                group = "other"
            rows.append({"run_id": run_id, "group": group, "value": value})
        box_df = pd.DataFrame(rows)
        if plot_boxplot(
            box_df,
            "Primary Metric Distribution",
            comparison_dir / "comparison_accuracy_box_plot.pdf",
            primary_metric,
        ):
            generated_paths.append(comparison_dir / "comparison_accuracy_box_plot.pdf")

    if significance_tests:
        sig_df = pd.DataFrame(significance_tests)
        sig_table_path = comparison_dir / "comparison_significance_table.pdf"
        plt.figure(figsize=(10, 0.6 + 0.3 * len(sig_df)))
        plt.axis("off")
        plt.table(
            cellText=np.round(sig_df[["proposed", "baseline", "z", "p", "diff"]].values, 4),
            colLabels=["proposed", "baseline", "z", "p", "diff"],
            cellLoc="center",
            loc="center",
        )
        plt.tight_layout()
        plt.savefig(sig_table_path)
        plt.close()
        generated_paths.append(sig_table_path)

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
