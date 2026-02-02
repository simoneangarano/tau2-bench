#!/usr/bin/env python3
"""
Check the percentage of simulations with had_unresponsive_period==True.

Usage:
    python scripts/check_unresponsive_rate.py <results.json> [<results.json> ...]

Example:
    python scripts/check_unresponsive_rate.py data/exp/*/results.json
"""

import argparse
import sys
from pathlib import Path

from tau2.data_model.simulation import Results


def check_unresponsive_rate(results_path: Path) -> dict:
    """
    Check the percentage of simulations with had_unresponsive_period==True.

    Returns:
        Dict with stats.
    """
    results = Results.load(results_path)

    total = 0
    with_ticks = 0
    unresponsive_count = 0

    for sim in results.simulations:
        total += 1
        if sim.ticks:
            with_ticks += 1
            if sim.info and sim.info.get("had_unresponsive_period"):
                unresponsive_count += 1

    rate = (unresponsive_count / with_ticks * 100) if with_ticks > 0 else 0

    return {
        "path": str(results_path),
        "total_simulations": total,
        "simulations_with_ticks": with_ticks,
        "unresponsive_count": unresponsive_count,
        "unresponsive_rate": rate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check the percentage of simulations with had_unresponsive_period==True."
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="Path(s) to results.json file(s)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output in CSV format",
    )

    args = parser.parse_args()

    # Validate files exist
    valid_files = []
    for f in args.files:
        if f.exists():
            valid_files.append(f)
        else:
            print(f"Warning: File not found: {f}", file=sys.stderr)

    if not valid_files:
        print("Error: No valid files found")
        sys.exit(1)

    results = []
    for f in valid_files:
        try:
            stats = check_unresponsive_rate(f)
            results.append(stats)
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)

    if args.csv:
        # CSV output
        print("file,total,with_ticks,unresponsive,rate_pct")
        for r in results:
            print(
                f"{r['path']},{r['total_simulations']},{r['simulations_with_ticks']},{r['unresponsive_count']},{r['unresponsive_rate']:.2f}"
            )
    else:
        # Table output
        print(f"{'File':<80} {'Total':>6} {'Ticks':>6} {'Unresp':>6} {'Rate':>8}")
        print("-" * 110)
        for r in results:
            # Shorten path for display
            short_path = str(r["path"])
            if len(short_path) > 78:
                short_path = "..." + short_path[-75:]
            print(
                f"{short_path:<80} {r['total_simulations']:>6} {r['simulations_with_ticks']:>6} {r['unresponsive_count']:>6} {r['unresponsive_rate']:>7.1f}%"
            )

        # Summary
        if len(results) > 1:
            total_sims = sum(r["simulations_with_ticks"] for r in results)
            total_unresp = sum(r["unresponsive_count"] for r in results)
            overall_rate = (total_unresp / total_sims * 100) if total_sims > 0 else 0
            print("-" * 110)
            print(
                f"{'TOTAL':<80} {sum(r['total_simulations'] for r in results):>6} {total_sims:>6} {total_unresp:>6} {overall_rate:>7.1f}%"
            )


if __name__ == "__main__":
    main()
