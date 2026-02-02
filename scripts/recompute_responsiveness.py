#!/usr/bin/env python3
"""
Recompute responsiveness info for all simulations in a directory.

This script finds all results.json files in subdirectories, recomputes the
responsiveness metrics using the fixed algorithm, and updates the simulation
info fields.

Usage:
    python scripts/recompute_responsiveness.py <directory>

Example:
    python scripts/recompute_responsiveness.py data/exp/tau-voice-experiments/experiment_2025_01_22_v4_regular
"""

import argparse
import sys
from pathlib import Path

from tau2.agent.base.streaming import compute_responsiveness_info
from tau2.data_model.simulation import Results


def find_results_files(directory: Path) -> list[Path]:
    """Find all results.json files in subdirectories."""
    return list(directory.rglob("results.json"))


def recompute_responsiveness_for_file(
    results_path: Path, dry_run: bool = False
) -> dict:
    """
    Recompute responsiveness info for all simulations in a results file.

    Returns:
        Dict with stats about what changed.
    """
    print(f"\nProcessing: {results_path}", flush=True)

    # Load results
    results = Results.load(results_path)

    stats = {
        "total_simulations": len(results.simulations),
        "simulations_with_ticks": 0,
        "had_unresponsive_changed": 0,
        "max_unresponded_changed": 0,
        "changes": [],
    }

    for sim in results.simulations:
        if not sim.ticks:
            continue

        stats["simulations_with_ticks"] += 1

        # Compute new responsiveness info
        new_info = compute_responsiveness_info(sim.ticks)

        # Get old values from sim.info
        if sim.info is None:
            sim.info = {}

        old_had_unresponsive = sim.info.get("had_unresponsive_period")
        old_max_unresponded = sim.info.get("max_unresponded_user_turns")

        new_had_unresponsive = new_info["had_unresponsive_period"]
        new_max_unresponded = new_info["max_unresponded_user_turns"]

        # Track changes
        had_unresponsive_changed = old_had_unresponsive != new_had_unresponsive
        max_unresponded_changed = old_max_unresponded != new_max_unresponded

        if had_unresponsive_changed:
            stats["had_unresponsive_changed"] += 1
        if max_unresponded_changed:
            stats["max_unresponded_changed"] += 1

        if had_unresponsive_changed or max_unresponded_changed:
            stats["changes"].append(
                {
                    "sim_id": sim.id,
                    "task_id": sim.task_id,
                    "old_had_unresponsive": old_had_unresponsive,
                    "new_had_unresponsive": new_had_unresponsive,
                    "old_max_unresponded": old_max_unresponded,
                    "new_max_unresponded": new_max_unresponded,
                }
            )

        # Update sim.info with new values
        sim.info["had_unresponsive_period"] = new_had_unresponsive
        sim.info["max_unresponded_user_turns"] = new_max_unresponded
        sim.info["total_user_turns"] = new_info["total_user_turns"]
        sim.info["total_agent_turns"] = new_info["total_agent_turns"]

    # Print summary for this file
    print(f"  Total simulations: {stats['total_simulations']}")
    print(f"  Simulations with ticks: {stats['simulations_with_ticks']}")
    print(f"  had_unresponsive_period changed: {stats['had_unresponsive_changed']}")
    print(f"  max_unresponded_user_turns changed: {stats['max_unresponded_changed']}")

    if stats["changes"]:
        print(f"  Changes:")
        for change in stats["changes"][:5]:  # Show first 5
            print(
                f"    {change['sim_id'][:8]}... had_unresponsive: {change['old_had_unresponsive']} -> {change['new_had_unresponsive']}, "
                f"max_unresponded: {change['old_max_unresponded']} -> {change['new_max_unresponded']}"
            )
        if len(stats["changes"]) > 5:
            print(f"    ... and {len(stats['changes']) - 5} more")

    # Save if not dry run
    if not dry_run and (
        stats["had_unresponsive_changed"] > 0 or stats["max_unresponded_changed"] > 0
    ):
        results.save(results_path)
        print(f"  ✓ Saved changes")
    elif dry_run:
        print(f"  (dry run - not saving)")
    else:
        print(f"  No changes needed")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Recompute responsiveness info for all simulations in a directory."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to search for results.json files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save changes, just show what would change",
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)

    # Find all results.json files
    results_files = find_results_files(args.directory)

    if not results_files:
        print(f"No results.json files found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(results_files)} results.json file(s)", flush=True)
    if args.dry_run:
        print("Running in dry-run mode (no changes will be saved)")

    # Process each file
    total_stats = {
        "files_processed": 0,
        "total_simulations": 0,
        "simulations_with_ticks": 0,
        "had_unresponsive_changed": 0,
        "max_unresponded_changed": 0,
    }

    for results_path in results_files:
        try:
            stats = recompute_responsiveness_for_file(
                results_path, dry_run=args.dry_run
            )
            total_stats["files_processed"] += 1
            total_stats["total_simulations"] += stats["total_simulations"]
            total_stats["simulations_with_ticks"] += stats["simulations_with_ticks"]
            total_stats["had_unresponsive_changed"] += stats["had_unresponsive_changed"]
            total_stats["max_unresponded_changed"] += stats["max_unresponded_changed"]
        except Exception as e:
            print(f"  Error processing {results_path}: {e}")

    # Print total summary
    print("\n" + "=" * 60)
    print("TOTAL SUMMARY")
    print("=" * 60)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total simulations: {total_stats['total_simulations']}")
    print(f"Simulations with ticks: {total_stats['simulations_with_ticks']}")
    print(f"had_unresponsive_period changed: {total_stats['had_unresponsive_changed']}")
    print(
        f"max_unresponded_user_turns changed: {total_stats['max_unresponded_changed']}"
    )


if __name__ == "__main__":
    main()
