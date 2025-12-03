#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

def load_metrics(log_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Load metrics from a training run's metrics.jsonl file."""
    metrics_file = log_dir / "metrics.jsonl"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    data = defaultdict(list)
    with open(metrics_file) as f:
        for line in f:
            entry = json.loads(line)
            step = entry.get("step", 0)
            for key, value in entry.items():
                if key != "step" and isinstance(value, (int, float)):
                    data[key].append((step, value))

    return dict(data)


def load_scores(log_dir: Path) -> List[Tuple[int, float]]:
    """Load episode scores from scores.jsonl file."""
    scores_file = log_dir / "scores.jsonl"
    if not scores_file.exists():
        return []

    scores = []
    with open(scores_file) as f:
        for line in f:
            entry = json.loads(line)
            step = entry.get("step", 0)
            score = entry.get("episode/score", 0)
            scores.append((step, score))

    return scores


def smooth(data: np.ndarray, window: int = 50) -> np.ndarray:
    """Apply smoothing to data using uniform filter."""
    if len(data) < window:
        return data
    return uniform_filter1d(data.astype(float), size=window, mode='nearest')


def extract_xy(data: List[Tuple[int, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract x (steps) and y (values) arrays from list of tuples."""
    if not data:
        return np.array([]), np.array([])
    steps, values = zip(*data)
    return np.array(steps), np.array(values)


def compute_statistics(values: np.ndarray, name: str) -> Dict:
    """Compute summary statistics for a metric."""
    if len(values) == 0:
        return {"name": name, "count": 0}

    return {
        "name": name,
        "count": len(values),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "final_mean": float(np.mean(values[-100:])) if len(values) >= 100 else float(np.mean(values)),
    }


def plot_comparison(
    runs_data: Dict[str, Dict],
    metric_key: str,
    ax: plt.Axes,
    title: str,
    ylabel: str,
    smooth_window: int = 50,
    show_raw: bool = False,
):
    """Plot a single metric comparison across runs."""
    for run_name, data in runs_data.items():
        if metric_key not in data["metrics"]:
            continue

        steps, values = extract_xy(data["metrics"][metric_key])
        if len(steps) == 0:
            continue

        color = data["color"]
        if show_raw:
            ax.plot(steps, values, alpha=0.2, color=color, linewidth=0.5)

        smoothed = smooth(values, smooth_window)
        ax.plot(steps, smoothed, label=run_name, color=color, linewidth=1.5)

    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def create_main_figure(runs_data: Dict[str, Dict], output_dir: Path):
    """Create the main comparison figure with key metrics."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)

    # Row 1: Episode Performance
    ax1 = fig.add_subplot(gs[0, 0])
    plot_comparison(runs_data, "episode/score", ax1, "Episode Score", "Score", smooth_window=20, show_raw=True)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_comparison(runs_data, "episode/length", ax2, "Episode Length", "Steps", smooth_window=20)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_comparison(runs_data, "train/ret", ax3, "Return (Imagined)", "Return")

    # Row 2: World Model Losses
    ax4 = fig.add_subplot(gs[1, 0])
    plot_comparison(runs_data, "train/loss/image", ax4, "Image Reconstruction Loss", "Loss")

    ax5 = fig.add_subplot(gs[1, 1])
    plot_comparison(runs_data, "train/loss/dyn", ax5, "Dynamics Loss (KL)", "Loss")

    ax6 = fig.add_subplot(gs[1, 2])
    plot_comparison(runs_data, "train/loss/rew", ax6, "Reward Prediction Loss", "Loss")

    # Row 3: Policy & Value Losses
    ax7 = fig.add_subplot(gs[2, 0])
    plot_comparison(runs_data, "train/loss/policy", ax7, "Policy Loss", "Loss")

    ax8 = fig.add_subplot(gs[2, 1])
    plot_comparison(runs_data, "train/loss/value", ax8, "Value Loss", "Loss")

    ax9 = fig.add_subplot(gs[2, 2])
    plot_comparison(runs_data, "train/ent/action", ax9, "Action Entropy", "Entropy")

    # Row 4: Training Dynamics
    ax10 = fig.add_subplot(gs[3, 0])
    plot_comparison(runs_data, "train/adv", ax10, "Advantage", "Advantage")

    ax11 = fig.add_subplot(gs[3, 1])
    plot_comparison(runs_data, "train/opt/grad_norm", ax11, "Gradient Norm", "Norm")

    ax12 = fig.add_subplot(gs[3, 2])
    plot_comparison(runs_data, "fps/train", ax12, "Training FPS", "FPS")

    plt.suptitle("DreamerV3 FPV Training Comparison - Main Metrics", fontsize=14, fontweight='bold')
    plt.savefig(output_dir / "comparison_main.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'comparison_main.png'}")


def create_auxiliary_figure(runs_data: Dict[str, Dict], output_dir: Path):
    """Create figure for auxiliary loss metrics (SkyDreamer-style)."""
    # Check if any run has aux metrics
    has_aux = any(
        "train/aux/state_mse" in data["metrics"] or "train/loss/aux_state" in data["metrics"]
        for data in runs_data.values()
    )

    if not has_aux:
        print("No auxiliary metrics found, skipping auxiliary figure.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Aux prediction errors
    plot_comparison(runs_data, "train/aux/state_mse", axes[0, 0], "State Prediction MSE", "MSE")
    plot_comparison(runs_data, "train/aux/goal_mse", axes[0, 1], "Goal Prediction MSE", "MSE")
    plot_comparison(runs_data, "train/aux/dist_error", axes[0, 2], "Distance Prediction Error", "Error (m)")

    # Aux losses
    plot_comparison(runs_data, "train/loss/aux_state", axes[1, 0], "Aux State Loss", "Loss")
    plot_comparison(runs_data, "train/loss/aux_goal", axes[1, 1], "Aux Goal Loss", "Loss")
    plot_comparison(runs_data, "train/loss/state", axes[1, 2], "State Reconstruction Loss", "Loss")

    plt.suptitle("Auxiliary State/Goal Prediction (SkyDreamer-style)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_auxiliary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'comparison_auxiliary.png'}")


def create_score_histogram(runs_data: Dict[str, Dict], output_dir: Path):
    """Create histogram comparing score distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution
    ax1 = axes[0]
    for run_name, data in runs_data.items():
        scores = data["scores"]
        if not scores:
            continue
        _, values = extract_xy(scores)
        ax1.hist(values, bins=50, alpha=0.5, label=run_name, color=data["color"])

    ax1.set_xlabel("Episode Score")
    ax1.set_ylabel("Count")
    ax1.set_title("Full Score Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Last 100 episodes
    ax2 = axes[1]
    for run_name, data in runs_data.items():
        scores = data["scores"]
        if not scores:
            continue
        _, values = extract_xy(scores)
        last_100 = values[-100:] if len(values) >= 100 else values
        ax2.hist(last_100, bins=30, alpha=0.5, label=f"{run_name} (n={len(last_100)})", color=data["color"])

    ax2.set_xlabel("Episode Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Last 100 Episodes Score Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_score_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'comparison_score_histogram.png'}")


def create_learning_curve_detail(runs_data: Dict[str, Dict], output_dir: Path):
    """Create detailed learning curve with confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for run_name, data in runs_data.items():
        scores = data["scores"]
        if not scores:
            continue

        steps, values = extract_xy(scores)
        color = data["color"]

        # Raw scores (very transparent)
        ax.scatter(steps, values, alpha=0.1, s=5, color=color)

        # Rolling statistics
        window = min(50, len(values) // 5) if len(values) > 10 else len(values)
        if window > 1:
            smoothed = smooth(values, window)
            ax.plot(steps, smoothed, label=f"{run_name} (smoothed)", color=color, linewidth=2)

            # Compute rolling std for confidence band
            rolling_std = np.array([
                np.std(values[max(0, i-window//2):i+window//2+1])
                for i in range(len(values))
            ])
            ax.fill_between(steps, smoothed - rolling_std, smoothed + rolling_std,
                          alpha=0.2, color=color)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Episode Score")
    ax.set_title("Learning Curve with Confidence Bands")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_learning_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'comparison_learning_curve.png'}")


def generate_report(runs_data: Dict[str, Dict], output_dir: Path):
    """Generate a text report with statistics."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DREAMERV3 FPV TRAINING ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    for run_name, data in runs_data.items():
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"RUN: {run_name}")
        report_lines.append(f"Path: {data['path']}")
        report_lines.append(f"{'='*40}")

        # Score statistics
        scores = data["scores"]
        if scores:
            steps, values = extract_xy(scores)
            stats = compute_statistics(values, "Episode Score")
            report_lines.append(f"\n EPISODE SCORES:")
            report_lines.append(f"   Total episodes: {stats['count']}")
            report_lines.append(f"   Training steps: {int(steps[-1]) if len(steps) > 0 else 0:,}")
            report_lines.append(f"   Score range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            report_lines.append(f"   Mean score: {stats['mean']:.2f} ± {stats['std']:.2f}")
            report_lines.append(f"   Final mean (last 100): {stats['final_mean']:.2f}")

            # Score milestones
            positive_episodes = np.sum(values > 0)
            report_lines.append(f"   Positive score episodes: {positive_episodes} ({100*positive_episodes/len(values):.1f}%)")

        # Key training metrics
        report_lines.append(f"\n KEY TRAINING METRICS (final values):")
        key_metrics = [
            ("train/loss/image", "Image Loss"),
            ("train/loss/dyn", "Dynamics Loss"),
            ("train/loss/rew", "Reward Loss"),
            ("train/loss/policy", "Policy Loss"),
            ("train/loss/value", "Value Loss"),
            ("train/ret", "Return"),
            ("train/ent/action", "Action Entropy"),
        ]

        for metric_key, metric_name in key_metrics:
            if metric_key in data["metrics"]:
                _, values = extract_xy(data["metrics"][metric_key])
                if len(values) > 0:
                    final = np.mean(values[-50:]) if len(values) >= 50 else np.mean(values)
                    report_lines.append(f"   {metric_name}: {final:.4f}")

        # Auxiliary metrics (if present)
        aux_metrics = [
            ("train/aux/state_mse", "State MSE"),
            ("train/aux/goal_mse", "Goal MSE"),
            ("train/aux/dist_error", "Distance Error"),
        ]

        has_aux = any(m in data["metrics"] for m, _ in aux_metrics)
        if has_aux:
            report_lines.append(f"\n AUXILIARY PREDICTIONS (SkyDreamer-style):")
            for metric_key, metric_name in aux_metrics:
                if metric_key in data["metrics"]:
                    _, values = extract_xy(data["metrics"][metric_key])
                    if len(values) > 0:
                        final = np.mean(values[-50:]) if len(values) >= 50 else np.mean(values)
                        report_lines.append(f"   {metric_name}: {final:.4f}")

    # Comparison summary
    if len(runs_data) > 1:
        report_lines.append(f"\n{'='*40}")
        report_lines.append("COMPARISON SUMMARY")
        report_lines.append(f"{'='*40}")

        # Compare final scores
        final_scores = {}
        for run_name, data in runs_data.items():
            if data["scores"]:
                _, values = extract_xy(data["scores"])
                final_scores[run_name] = np.mean(values[-100:]) if len(values) >= 100 else np.mean(values)

        if final_scores:
            best_run = max(final_scores, key=final_scores.get)
            report_lines.append(f"\n Best performing run: {best_run}")
            report_lines.append(f"   Final mean score: {final_scores[best_run]:.2f}")

            for run_name, score in final_scores.items():
                if run_name != best_run:
                    diff = final_scores[best_run] - score
                    report_lines.append(f"   vs {run_name}: +{diff:.2f} better")

    report_lines.append("\n" + "=" * 80)

    # Write report
    report_text = "\n".join(report_lines)
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DreamerV3 FPV training runs")
    parser.add_argument("--runs", nargs="+", help="Paths to log directories")
    parser.add_argument("--output", type=str, default="./analysis_output", help="Output directory for figures")
    parser.add_argument("--names", nargs="+", default=None, help="Custom names for runs")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define colors for runs
    colors = plt.cm.tab10.colors

    # Load data for each run
    runs_data = {}
    for i, run_path in enumerate(args.runs):
        log_dir = Path(run_path)
        if not log_dir.exists():
            print(f"Warning: Run directory not found: {log_dir}")
            continue

        run_name = args.names[i] if args.names and i < len(args.names) else log_dir.name

        print(f"Loading {run_name}...")
        try:
            metrics = load_metrics(log_dir)
            scores = load_scores(log_dir)
            runs_data[run_name] = {
                "path": str(log_dir),
                "metrics": metrics,
                "scores": scores,
                "color": colors[i % len(colors)],
            }
            print(f"  Loaded {len(metrics)} metric types, {len(scores)} episodes")
        except Exception as e:
            print(f"  Error loading: {e}")

    if not runs_data:
        print("No valid runs found!")
        return

    # Generate figures
    print("\nGenerating figures...")
    create_main_figure(runs_data, output_dir)
    create_auxiliary_figure(runs_data, output_dir)
    create_score_histogram(runs_data, output_dir)
    create_learning_curve_detail(runs_data, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(runs_data, output_dir)

    print(f"\n Analysis complete! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
