#!/usr/bin/env python3
"""
Cleanup script for llm_debug folders.

For each llm_debug folder, keeps only the last file for each LLM call type
(user_streaming_response, interruption_decision, backchannel_decision, etc.).

Supports two directory structures:
1. Old: llm_debug/sim_<uuid>/*.json
2. New: tasks/task_N/sim_<uuid>/llm_debug/*.json
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path


def get_call_type(filename: str) -> str | None:
    """Extract the call type from a filename.

    Expected format: YYYYMMDD_HHMMSS_mmm_<call_type>_<uuid>.json
    """
    # Match pattern like: 20260121_083146_954_user_streaming_response_58bf5877.json
    match = re.match(r"\d{8}_\d{6}_\d{3}_(.+)_[a-f0-9]{8}\.json$", filename)
    if match:
        return match.group(1)
    return None


def process_llm_debug_folder(
    llm_debug_folder: Path, dry_run: bool = True, verbose: bool = False
) -> tuple[int, int]:
    """Process a single llm_debug folder.

    Returns:
        Tuple of (files_deleted, files_kept)
    """
    # Group files by call type
    files_by_type: dict[str, list[str]] = defaultdict(list)

    for file in llm_debug_folder.iterdir():
        if not file.is_file() or not file.suffix == ".json":
            continue

        call_type = get_call_type(file.name)
        if call_type:
            files_by_type[call_type].append(file.name)

    files_to_delete = []
    files_to_keep = []

    for call_type, files in files_by_type.items():
        # Sort by filename (which is chronological due to timestamp prefix)
        sorted_files = sorted(files)

        # Keep the last one, delete the rest
        if sorted_files:
            files_to_keep.append(sorted_files[-1])
            files_to_delete.extend(sorted_files[:-1])

    # Actually delete files if not dry run
    for filename in files_to_delete:
        file_path = llm_debug_folder / filename
        if dry_run:
            if verbose:
                print(f"    Would delete: {filename}")
        else:
            file_path.unlink()

    return len(files_to_delete), len(files_to_keep)


def find_llm_debug_folders(root_path: Path) -> list[Path]:
    """Recursively find all llm_debug folders under root_path."""
    llm_debug_folders = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        path = Path(dirpath)
        if path.name == "llm_debug":
            llm_debug_folders.append(path)

    return sorted(llm_debug_folders)


def process_directory(
    root_path: Path, dry_run: bool = True, verbose: bool = False
) -> None:
    """Process all llm_debug folders under root_path."""

    if not root_path.exists():
        print(f"Error: Path does not exist: {root_path}")
        return

    # Find all llm_debug folders
    llm_debug_folders = find_llm_debug_folders(root_path)

    print(f"Found {len(llm_debug_folders)} llm_debug folders under {root_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'DELETING FILES'}")
    print("-" * 60)

    total_deleted = 0
    total_kept = 0

    for llm_debug_folder in llm_debug_folders:
        deleted, kept = process_llm_debug_folder(llm_debug_folder, dry_run, verbose)
        total_deleted += deleted
        total_kept += kept

        # Show relative path for readability
        rel_path = llm_debug_folder.relative_to(root_path)
        if deleted > 0:
            print(f"{rel_path}: delete {deleted}, keep {kept}")

    print("-" * 60)
    print(f"Total: {total_deleted} files to delete, {total_kept} files to keep")

    if dry_run:
        print("\nThis was a dry run. Use --execute to actually delete files.")


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup llm_debug folders by keeping only the last file for each call type per simulation."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to directory containing llm_debug folders (will search recursively)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default is dry run)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show files that would be deleted in dry run mode",
    )

    args = parser.parse_args()

    root_path = Path(args.path).resolve()
    process_directory(root_path, dry_run=not args.execute, verbose=args.verbose)


if __name__ == "__main__":
    main()
