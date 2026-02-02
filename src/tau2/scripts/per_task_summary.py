#!/usr/bin/env python3
"""
Generate per-task reward summary table for different domains.

Usage:
    python src/tau2/scripts/per_task_summary.py --domain retail
    python src/tau2/scripts/per_task_summary.py --domain telecom
    python src/tau2/scripts/per_task_summary.py --domain airline
"""

import argparse
import json
from pathlib import Path


def get_rewards(data: dict) -> dict:
    """Extract task_id -> reward mapping from results data."""
    rewards = {}
    for s in data.get("simulations", []):
        task_id = s.get("task_id")
        ri = s.get("reward_info")
        reward = ri.get("reward", 0) if ri else 0
        rewards[str(task_id)] = reward
    return rewards


def load_results(path: str) -> dict:
    """Load results.json from path."""
    with open(path) as f:
        return json.load(f)


def get_domain_config(domain: str, base: Path) -> tuple[dict, str]:
    """Get experiment paths and output filename for a domain."""
    if domain == "retail":
        experiments = {
            "GPT-4.1 (text)": base / "retail_gpt41_text/results.json",
            "GPT-5.2-med (text)": base / "retail_gpt52medium_text/results.json",
            "Gem ctrl": base
            / "victor/experiment_2025_01_22_v4_control/retail_control_gemini/results.json",
            "OAI ctrl": base
            / "oai_retailctrl_rerun/retail_control_openai/results.json",
            "XAI ctrl": base
            / "victor/experiment_2025_01_22_v4_control/retail_control_xai/results.json",
            "Gem reg": base
            / "victor/experiment_2025_01_22_v4_regular/retail_regular_gemini/results.json",
            "OAI reg": base
            / "victor/experiment_2025_01_22_v4_regular/retail_regular_openai/results.json",
            "XAI reg": base
            / "victor/experiment_2025_01_22_v4_regular/retail_regular_xai/results.json",
        }
        output_file = "retail_per_task_summary.txt"
    elif domain == "telecom":
        experiments = {
            "GPT-4.1 (text)": base / "telecom_gpt41_text/results.json",
            "GPT-5.2-med (text)": base / "telecom_gpt52medium_text/results.json",
            "Gem ctrl": base
            / "exp_20260124_telecom_control_1/telecom_control_gemini/results.json",
            "OAI ctrl": base
            / "exp_20260124_telecom_control_1/telecom_control_openai/results.json",
            "XAI ctrl": base
            / "exp_20260124_telecom_control_2/telecom_control_xai/results.json",
            "Gem reg": base / "exp_telecom_regular/telecom_regular_gemini/results.json",
            "OAI reg": base / "exp_telecom_regular/telecom_regular_openai/results.json",
            "XAI reg": base / "exp_telecom_regular/telecom_regular_xai/results.json",
        }
        output_file = "telecom_per_task_summary.txt"
    elif domain == "airline":
        # Airline audio is on gdrive
        gdrive = Path(
            "/Users/keshav@sierra.ai/Library/CloudStorage/GoogleDrive-keshav@sierra.ai/.shortcut-targets-by-id/1ikm0YXSXWy9SRV2LyMKuUkAKHxTlkefj/tau-voice-gdrive"
        )
        experiments = {
            "GPT-4.1 (text)": base / "airline_gpt41_text/results.json",
            "GPT-5.2-med (text)": base / "airline_gpt52medium_text/results.json",
            "Gem ctrl": gdrive
            / "2026_01_25_v4_airline/airline_control_gemini/results.json",
            "OAI ctrl": gdrive
            / "2026_01_25_v4_airline/airline_control_openai/results.json",
            "XAI ctrl": gdrive
            / "2026_01_25_v4_airline/airline_control_xai/results.json",
            "Gem reg": gdrive
            / "2026_01_25_v4_airline/airline_regular_gemini/results.json",
            "OAI reg": gdrive
            / "2026_01_25_v4_airline/airline_regular_openai/results.json",
            "XAI reg": gdrive
            / "2026_01_25_v4_airline/airline_regular_xai/results.json",
        }
        output_file = "airline_per_task_summary.txt"
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return experiments, output_file


def generate_summary(domain: str):
    base = Path("tmp/qa_results")

    experiments, output_file = get_domain_config(domain, base)

    rewards = {}
    for name, path in experiments.items():
        if path.exists():
            rewards[name] = get_rewards(load_results(path))
        else:
            print(f"Warning: {path} not found")
            rewards[name] = {}

    # Get all task IDs - preserve order from first results file (reflects run order)
    all_tasks = []
    seen = set()
    # Use the first text experiment as the reference for ordering
    first_exp_path = list(experiments.values())[0]
    if first_exp_path.exists():
        with open(first_exp_path) as f:
            first_data = json.load(f)
        for s in first_data.get("simulations", []):
            tid = str(s.get("task_id"))
            if tid not in seen:
                all_tasks.append(tid)
                seen.add(tid)

    # Add any tasks from other experiments that weren't in the first
    for r in rewards.values():
        for tid in r.keys():
            if tid not in seen:
                all_tasks.append(tid)
                seen.add(tid)

    # Column widths
    col_names = list(experiments.keys())
    col_widths = {name: max(len(name), 5) for name in col_names}

    # Identify column types
    text_cols = [n for n in col_names if "text" in n.lower()]
    ctrl_cols = [n for n in col_names if "ctrl" in n.lower()]
    reg_cols = [n for n in col_names if "reg" in n.lower()]
    audio_cols = ctrl_cols + reg_cols  # All non-text are audio
    if not audio_cols:
        audio_cols = [n for n in col_names if "audio" in n.lower()]

    # Create table
    lines = []
    lines.append(f"# Per-Task Reward Summary - {domain.capitalize()} Domain")
    lines.append("")

    # Determine task ID width based on domain
    max_task_len = max(len(str(t)) for t in all_tasks) if all_tasks else 7
    task_width = min(max_task_len, 80)  # Cap at 80 chars

    # Header
    header = f"| {'Task ID':<{task_width}} |"
    separator = "|" + "-" * (task_width + 2) + "|"
    for name in col_names:
        w = col_widths[name]
        header += f" {name:^{w}} |"
        separator += "-" * (w + 2) + "|"
    header += " Label                    |"
    separator += "--------------------------|"
    # Add second label column for ctrl vs reg comparison if both exist
    has_ctrl_reg = len(ctrl_cols) > 0 and len(reg_cols) > 0
    if has_ctrl_reg:
        header += " Ctrl vs Reg             |"
        separator += "-------------------------|"
    lines.append(header)
    lines.append(separator)

    # Counters for labels (separate for label1 and label2)
    label1_counts = {}
    label2_counts = {}

    # Data rows
    for task_id in all_tasks:
        # Truncate long task IDs for display
        display_id = str(task_id)[:task_width]
        row = f"| {display_id:<{task_width}} |"
        for name in col_names:
            r = rewards[name].get(task_id, "-")
            w = col_widths[name]
            val = f"{r:.1f}" if isinstance(r, float) else str(r)
            row += f" {val:^{w}} |"

        # Calculate label (text vs control audio only)
        text_passed = sum(1 for n in text_cols if rewards[n].get(task_id, 0) == 1.0)
        # For first label, only compare against control audio (not regular)
        ctrl_audio_cols = ctrl_cols if ctrl_cols else audio_cols
        ctrl_audio_passed = sum(
            1 for n in ctrl_audio_cols if rewards[n].get(task_id, 0) == 1.0
        )

        # Check for XAI-only pass (XAI ctrl passes, all others fail)
        xai_ctrl_cols = [n for n in ctrl_audio_cols if "XAI" in n or "xai" in n.lower()]
        xai_ctrl_passed = sum(
            1 for n in xai_ctrl_cols if rewards[n].get(task_id, 0) == 1.0
        )
        non_xai_ctrl = [n for n in ctrl_audio_cols if n not in xai_ctrl_cols]
        non_xai_ctrl_passed = sum(
            1 for n in non_xai_ctrl if rewards[n].get(task_id, 0) == 1.0
        )

        label = ""
        if text_passed == len(text_cols):  # Both text passed
            if ctrl_audio_passed == 0:
                label = "TEXT_ONLY"
            elif ctrl_audio_passed == 1:
                label = "TEXT+1_CTRL"
            elif ctrl_audio_passed == len(ctrl_audio_cols):
                label = "ALL_PASS"
            else:
                label = f"TEXT+{ctrl_audio_passed}_CTRL"
        elif text_passed == 0 and ctrl_audio_passed == 0:
            label = "ALL_FAIL"
        elif text_passed == 0 and xai_ctrl_passed > 0 and non_xai_ctrl_passed == 0:
            label = "XAI_ONLY"
        else:
            label = ""

        label1_counts[label] = label1_counts.get(label, 0) + 1
        row += f" {label:<24} |"

        # Second label: ctrl vs reg comparison
        if has_ctrl_reg:
            ctrl_passed = sum(1 for n in ctrl_cols if rewards[n].get(task_id, 0) == 1.0)
            reg_passed = sum(1 for n in reg_cols if rewards[n].get(task_id, 0) == 1.0)
            # Only label if reg has data for this task
            reg_has_data = any(rewards[n].get(task_id) is not None for n in reg_cols)

            label2 = ""
            if reg_has_data:
                diff = ctrl_passed - reg_passed
                if diff >= 3:
                    label2 = "CTRL+3"
                elif diff == 2:
                    label2 = "CTRL+2"
                elif diff <= -3:
                    label2 = "REG+3"
                elif diff == -2:
                    label2 = "REG+2"

            label2_counts[label2] = label2_counts.get(label2, 0) + 1
            row += f" {label2:<23} |"

        lines.append(row)

    # Summary
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model                 | Pass Rate       |")
    lines.append("|-----------------------|-----------------|")
    for name in col_names:
        r = rewards[name]
        if r:
            passed = sum(1 for v in r.values() if v == 1.0)
            total = len(r)
            pct = passed / total * 100 if total else 0
            lines.append(f"| {name:<21} | {passed:>3}/{total:<3} ({pct:>5.1f}%) |")

    # Label 1 distribution (text vs control audio)
    lines.append("")
    lines.append("## Label Distribution (Text vs Control)")
    lines.append("")
    lines.append("| Label                    | Count |")
    lines.append("|--------------------------|-------|")
    for label in [
        "TEXT_ONLY",
        "TEXT+1_CTRL",
        "TEXT+2_CTRL",
        "TEXT+3_CTRL",
        "ALL_PASS",
        "ALL_FAIL",
        "XAI_ONLY",
        "",
    ]:
        if label in label1_counts:
            display = label if label else "(other)"
            lines.append(f"| {display:<24} | {label1_counts[label]:>5} |")

    # Label 2 distribution (ctrl vs reg) - only if applicable
    if has_ctrl_reg:
        lines.append("")
        lines.append("## Label Distribution (Ctrl vs Reg)")
        lines.append("")
        lines.append("| Label                    | Count |")
        lines.append("|--------------------------|-------|")
        for label in ["CTRL+3", "CTRL+2", "REG+2", "REG+3", ""]:
            if label in label2_counts:
                display = label if label else "(no diff >= 2)"
                lines.append(f"| {display:<24} | {label2_counts[label]:>5} |")

    output = "\n".join(lines)

    # Write to file
    output_path = base / output_file
    with open(output_path, "w") as f:
        f.write(output)

    print(f"Saved to {output_path}")
    print()
    print("Summary:")
    for name in col_names:
        r = rewards[name]
        if r:
            passed = sum(1 for v in r.values() if v == 1.0)
            total = len(r)
            pct = passed / total * 100 if total else 0
            print(f"  {name}: {passed}/{total} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate per-task reward summary")
    parser.add_argument(
        "--domain",
        type=str,
        default="retail",
        choices=["retail", "telecom", "airline"],
        help="Domain to generate summary for",
    )
    args = parser.parse_args()

    generate_summary(args.domain)


if __name__ == "__main__":
    main()
