#!/usr/bin/env python3
"""
Cleanup script for audio folders.

Deletes user.wav and assistant.wav files, keeping only both.wav.
This reduces storage by removing redundant individual audio tracks.

Directory structure expected:
    experiment_root/
    ├── retail_control_openai/
    │   └── tasks/
    │       └── task_N/
    │           └── sim_<uuid>/
    │               └── audio/
    │                   ├── assistant_labels.txt  (kept)
    │                   ├── assistant.wav         (deleted)
    │                   ├── both.wav              (kept)
    │                   ├── user_labels.txt       (kept)
    │                   └── user.wav              (deleted)
"""

import argparse
import os
from pathlib import Path

# Files to delete
FILES_TO_DELETE = {"user.wav", "assistant.wav"}


def process_audio_folder(
    audio_folder: Path, dry_run: bool = True, verbose: bool = False
) -> tuple[int, int]:
    """Process a single audio folder.

    Returns:
        Tuple of (files_deleted, bytes_freed)
    """
    files_deleted = 0
    bytes_freed = 0

    for filename in FILES_TO_DELETE:
        file_path = audio_folder / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            if dry_run:
                if verbose:
                    print(
                        f"    Would delete: {filename} ({file_size / 1024 / 1024:.2f} MB)"
                    )
            else:
                file_path.unlink()
            files_deleted += 1
            bytes_freed += file_size

    return files_deleted, bytes_freed


def find_audio_folders(root_path: Path) -> list[Path]:
    """Recursively find all audio folders under root_path."""
    audio_folders = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        path = Path(dirpath)
        if path.name == "audio":
            # Verify it has both.wav (to avoid deleting from wrong folders)
            if (path / "both.wav").exists():
                audio_folders.append(path)

    return sorted(audio_folders)


def format_size(bytes_count: int) -> str:
    """Format bytes as human-readable size."""
    if bytes_count >= 1024 * 1024 * 1024:
        return f"{bytes_count / 1024 / 1024 / 1024:.2f} GB"
    elif bytes_count >= 1024 * 1024:
        return f"{bytes_count / 1024 / 1024:.2f} MB"
    elif bytes_count >= 1024:
        return f"{bytes_count / 1024:.2f} KB"
    else:
        return f"{bytes_count} bytes"


def process_directory(
    root_path: Path, dry_run: bool = True, verbose: bool = False
) -> None:
    """Process all audio folders under root_path."""

    if not root_path.exists():
        print(f"Error: Path does not exist: {root_path}")
        return

    # Find all audio folders
    audio_folders = find_audio_folders(root_path)

    print(f"Found {len(audio_folders)} audio folders under {root_path}")
    print(f"Mode: {'DRY RUN' if dry_run else 'DELETING FILES'}")
    print(f"Files to remove: {', '.join(sorted(FILES_TO_DELETE))}")
    print("-" * 60)

    total_deleted = 0
    total_bytes = 0

    for audio_folder in audio_folders:
        deleted, bytes_freed = process_audio_folder(audio_folder, dry_run, verbose)
        total_deleted += deleted
        total_bytes += bytes_freed

        # Show relative path for readability
        rel_path = audio_folder.relative_to(root_path)
        if verbose and deleted > 0:
            print(f"{rel_path}: {deleted} files ({format_size(bytes_freed)})")

    print("-" * 60)
    print(f"Total: {total_deleted} files to delete, {format_size(total_bytes)} to free")

    if dry_run:
        print("\nThis was a dry run. Use --execute to actually delete files.")


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup audio folders by removing user.wav and assistant.wav (keeping both.wav)."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to experiment directory (will search recursively for audio folders)",
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
        help="Show each folder being processed",
    )

    args = parser.parse_args()

    root_path = Path(args.path).resolve()
    process_directory(root_path, dry_run=not args.execute, verbose=args.verbose)


if __name__ == "__main__":
    main()
