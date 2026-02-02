#!/usr/bin/env python3
"""
Check experiment directory integrity.

This script validates that experiment directories have:
1. A results.json file in each experiment subdirectory
2. All tasks in results.json have the expected number of trials (ERROR if missing/incomplete)
3. For each simulation in results.json, a corresponding sim_<id> directory exists
4. Each sim directory contains audio/both.wav (ERROR if missing)
5. Each sim directory contains task.log (WARNING if missing)

Usage:
    python scripts/check_experiment_integrity.py /path/to/experiment/directory
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tau2.data_model.simulation import Results


def check_experiment_run(run_dir: Path) -> dict:
    """
    Check a single experiment run directory (one that contains results.json).

    Returns a dict with:
        - errors: list of error messages
        - warnings: list of warning messages
        - stats: dict with counts
    """
    errors = []
    warnings = []
    stats = {
        "total_tasks": 0,
        "tasks_with_all_trials": 0,
        "expected_trials_per_task": 0,
        "total_simulations": 0,
        "found_sim_dirs": 0,
        "has_both_wav": 0,
        "has_task_log": 0,
    }

    results_path = run_dir / "results.json"

    # Check results.json exists
    if not results_path.exists():
        errors.append(f"Missing results.json in {run_dir}")
        return {"errors": errors, "warnings": warnings, "stats": stats}

    # Load results
    try:
        results = Results.load(results_path)
    except Exception as e:
        errors.append(f"Failed to load results.json in {run_dir}: {e}")
        return {"errors": errors, "warnings": warnings, "stats": stats}

    # Check that all tasks have the expected number of trials
    expected_trials = results.info.num_trials
    stats["expected_trials_per_task"] = expected_trials
    stats["total_tasks"] = len(results.tasks)

    # Count simulations per task
    sims_per_task: dict[str, int] = defaultdict(int)
    for sim in results.simulations:
        sims_per_task[sim.task_id] += 1

    # Check each task has the expected number of simulations
    task_ids_in_results = {task.id for task in results.tasks}
    for task in results.tasks:
        task_id = task.id
        sim_count = sims_per_task.get(task_id, 0)
        if sim_count == 0:
            errors.append(
                f"Task {task_id} has no simulations (expected {expected_trials})"
            )
        elif sim_count < expected_trials:
            errors.append(
                f"Task {task_id} has {sim_count}/{expected_trials} simulations (incomplete)"
            )
        elif sim_count > expected_trials:
            warnings.append(
                f"Task {task_id} has {sim_count}/{expected_trials} simulations (extra trials)"
            )
        else:
            stats["tasks_with_all_trials"] += 1

    # Check for simulations with task_ids not in the task list
    for task_id in sims_per_task:
        if task_id not in task_ids_in_results:
            warnings.append(
                f"Found {sims_per_task[task_id]} simulation(s) for task {task_id} "
                "which is not in the tasks list"
            )

    tasks_dir = run_dir / "tasks"
    if not tasks_dir.exists():
        errors.append(f"Missing tasks/ directory in {run_dir}")
        return {"errors": errors, "warnings": warnings, "stats": stats}

    # Check each simulation
    for sim in results.simulations:
        stats["total_simulations"] += 1
        sim_id = sim.id
        task_id = sim.task_id

        # Find the task directory
        # Task directories can be named task_<task_id> or task_<task_id_with_special_chars>
        task_dir = None
        for candidate in tasks_dir.iterdir():
            if candidate.is_dir():
                # Match task_<number> or task_<full_id>
                if (
                    candidate.name == f"task_{task_id}"
                    or candidate.name
                    == f"task_{task_id.split('_')[0] if '_' in str(task_id) else task_id}"
                ):
                    task_dir = candidate
                    break
                # Also check if task_id is just a number
                if str(task_id).isdigit() and candidate.name == f"task_{task_id}":
                    task_dir = candidate
                    break

        # If not found by task_id, search for the sim directory directly
        sim_dir = None
        if task_dir:
            sim_dir = task_dir / f"sim_{sim_id}"

        # If task_dir not found, search all task directories for sim_<id>
        if sim_dir is None or not sim_dir.exists():
            for task_candidate in tasks_dir.iterdir():
                if task_candidate.is_dir():
                    potential_sim = task_candidate / f"sim_{sim_id}"
                    if potential_sim.exists():
                        sim_dir = potential_sim
                        task_dir = task_candidate
                        break

        if sim_dir is None or not sim_dir.exists():
            errors.append(
                f"Missing sim directory for simulation {sim_id} (task {task_id})"
            )
            continue

        stats["found_sim_dirs"] += 1

        # Check for audio/both.wav
        audio_dir = sim_dir / "audio"
        both_wav = audio_dir / "both.wav"
        if not both_wav.exists():
            errors.append(f"Missing audio/both.wav for sim_{sim_id}")
        else:
            stats["has_both_wav"] += 1

        # Check for task.log
        task_log = sim_dir / "task.log"
        if not task_log.exists():
            warnings.append(f"Missing task.log for sim_{sim_id}")
        else:
            stats["has_task_log"] += 1

    return {"errors": errors, "warnings": warnings, "stats": stats}


def find_experiment_runs(base_dir: Path) -> list[Path]:
    """
    Find all directories containing results.json.
    """
    runs = []
    for results_file in base_dir.rglob("results.json"):
        runs.append(results_file.parent)
    return runs


def main():
    parser = argparse.ArgumentParser(description="Check experiment directory integrity")
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all warnings (by default only errors are shown)",
    )
    parser.add_argument(
        "--summary-only",
        "-s",
        action="store_true",
        help="Only show summary, not individual errors",
    )

    args = parser.parse_args()

    base_dir = args.directory.resolve()
    if not base_dir.exists():
        print(f"Error: Directory does not exist: {base_dir}")
        sys.exit(1)

    print("=" * 80)
    print("EXPERIMENT INTEGRITY CHECK")
    print(f"Directory: {base_dir}")
    print("=" * 80)

    # Find all experiment runs
    runs = find_experiment_runs(base_dir)

    if not runs:
        print("\nNo experiment runs found (no results.json files)")
        sys.exit(1)

    print(f"\nFound {len(runs)} experiment run(s) with results.json\n")

    total_errors = 0
    total_warnings = 0
    all_stats = defaultdict(int)

    for run_dir in sorted(runs):
        rel_path = run_dir.relative_to(base_dir)
        result = check_experiment_run(run_dir)

        errors = result["errors"]
        warnings = result["warnings"]
        stats = result["stats"]

        # Aggregate stats
        for k, v in stats.items():
            all_stats[k] += v

        total_errors += len(errors)
        total_warnings += len(warnings)

        # Print run status
        if errors:
            status = "✗ ERRORS"
        elif warnings:
            status = "⚠ WARNINGS"
        else:
            status = "✓ OK"

        print(f"{status} {rel_path}")
        print(
            f"    Tasks: {stats['total_tasks']} "
            f"(complete: {stats['tasks_with_all_trials']}/{stats['total_tasks']}, "
            f"expected trials: {stats['expected_trials_per_task']})"
        )
        print(
            f"    Simulations: {stats['total_simulations']}, "
            f"Found dirs: {stats['found_sim_dirs']}, "
            f"audio/both.wav: {stats['has_both_wav']}, "
            f"task.log: {stats['has_task_log']}"
        )

        if not args.summary_only:
            for err in errors:
                print(f"    ERROR: {err}")
            if args.verbose:
                for warn in warnings:
                    print(f"    WARNING: {warn}")

        print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total experiment runs: {len(runs)}")
    print(f"Total tasks: {all_stats['total_tasks']}")
    print(f"Tasks with all trials complete: {all_stats['tasks_with_all_trials']}")
    print(f"Total simulations in results.json: {all_stats['total_simulations']}")
    print(f"Simulation directories found: {all_stats['found_sim_dirs']}")
    print(f"With audio/both.wav: {all_stats['has_both_wav']}")
    print(f"With task.log: {all_stats['has_task_log']}")
    print()
    print(f"Total ERRORS: {total_errors}")
    print(f"Total WARNINGS: {total_warnings}")

    if total_errors > 0:
        print("\n✗ INTEGRITY CHECK FAILED")
        sys.exit(1)
    elif total_warnings > 0:
        print("\n⚠ INTEGRITY CHECK PASSED WITH WARNINGS")
        sys.exit(0)
    else:
        print("\n✓ INTEGRITY CHECK PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
