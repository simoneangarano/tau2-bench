#!/usr/bin/env python3
"""
Test maximum concurrency for tau2 experiments.

This script runs simulations concurrently to find the maximum concurrency level
that works without errors (rate limits, connection failures, etc.) for a given
provider/domain configuration.

Strategy:
1. Exponential search: Start at 1, double until failures occur
2. Binary search: Narrow down to find exact maximum
3. Uses short simulations (30s default) to make tests faster

Usage:
    # Test REST API concurrency (default)
    python scripts/test_concurrency.py --domain airline

    # Test OpenAI Realtime API (audio-native mode)
    python scripts/test_concurrency.py --domain airline --audio-native --provider openai

    # Test Gemini Live on retail domain with custom settings
    python scripts/test_concurrency.py --domain retail --audio-native --provider gemini --max-test 32

    # Test with specific model
    python scripts/test_concurrency.py --domain airline --audio-native --provider openai --model gpt-4o-realtime-preview

Examples:
    # Quick test to find OpenAI Realtime concurrency limit
    python scripts/test_concurrency.py --domain airline --audio-native --provider openai --test-duration 30

    # More thorough test with longer simulations
    python scripts/test_concurrency.py --domain airline --audio-native --provider openai --test-duration 60 --max-test 64

    # Test REST API mode
    python scripts/test_concurrency.py --domain airline --llm-agent openai/gpt-4o

    # Save results to specific file
    python scripts/test_concurrency.py --domain airline --audio-native --output results.json

    # Don't save results
    python scripts/test_concurrency.py --domain airline --audio-native --no-save
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from tau2.config import DEFAULT_AUDIO_NATIVE_MODELS
from tau2.data_model.simulation import AudioNativeConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.run import get_tasks, run_task


@dataclass
class ConcurrencyTestResult:
    """Result of testing a specific concurrency level."""

    concurrency: int
    successful: int
    failed: int
    errors: list[str] = field(default_factory=list)
    error_categories: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.successful + self.failed
        return self.successful / total if total > 0 else 0.0

    @property
    def all_successful(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "concurrency": self.concurrency,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "errors": self.errors,
            "error_categories": self.error_categories,
            "duration_seconds": round(self.duration_seconds, 2),
        }


@dataclass
class ConcurrencyTestReport:
    """Full report of concurrency testing."""

    # Configuration
    domain: str
    mode: str  # "audio-native" or "rest-api"
    provider: Optional[str]
    model: Optional[str]
    llm_agent: Optional[str]
    llm_user: str
    task_id: str
    test_duration_seconds: int
    max_test_concurrency: int
    seed: int

    # Results
    results: list[ConcurrencyTestResult]
    max_successful_concurrency: int
    first_failed_concurrency: Optional[int]

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "config": {
                "domain": self.domain,
                "mode": self.mode,
                "provider": self.provider,
                "model": self.model,
                "llm_agent": self.llm_agent,
                "llm_user": self.llm_user,
                "task_id": self.task_id,
                "test_duration_seconds": self.test_duration_seconds,
                "max_test_concurrency": self.max_test_concurrency,
                "seed": self.seed,
            },
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "max_successful_concurrency": self.max_successful_concurrency,
                "first_failed_concurrency": self.first_failed_concurrency,
            },
            "timestamp": self.timestamp,
        }

    def save(self, output_path: Path) -> None:
        """Save report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"\n📄 Results saved to: {output_path}")


def categorize_error(error_msg: str) -> str:
    """Categorize an error message into common failure types."""
    error_lower = error_msg.lower()

    if "rate" in error_lower and "limit" in error_lower:
        return "rate_limit"
    elif "429" in error_msg:
        return "rate_limit"
    elif "connection" in error_lower and (
        "closed" in error_lower or "refused" in error_lower
    ):
        return "connection_closed"
    elif "timeout" in error_lower:
        return "timeout"
    elif "websocket" in error_lower:
        return "websocket_error"
    elif "quota" in error_lower:
        return "quota_exceeded"
    elif "capacity" in error_lower or "overloaded" in error_lower:
        return "server_overloaded"
    else:
        return "other"


def test_concurrency_level(
    domain: str,
    task,
    concurrency: int,
    audio_native_config: Optional[AudioNativeConfig],
    llm_agent: str,
    llm_user: str,
    max_steps: int,
    seed: int = 42,
) -> ConcurrencyTestResult:
    """
    Test a specific concurrency level by running N simulations in parallel.

    Args:
        domain: Domain to test
        task: Task to run
        concurrency: Number of concurrent simulations
        audio_native_config: Audio-native config (None for REST API mode)
        llm_agent: LLM model for agent
        llm_user: LLM model for user
        max_steps: Max steps for simulation
        seed: Base random seed

    Returns:
        ConcurrencyTestResult with success/failure counts and errors
    """
    successful = 0
    failed = 0
    errors = []

    def run_single(idx: int) -> tuple[bool, Optional[str]]:
        """Run a single simulation."""
        try:
            # Use different seed for each concurrent run
            run_seed = seed + idx

            _ = run_task(
                domain=domain,
                task=task,
                agent="llm",
                user="user_sim",
                llm_agent=llm_agent,
                llm_args_agent={},
                llm_user=llm_user,
                llm_args_user={},
                max_steps=max_steps,
                max_errors=10,
                evaluation_type=EvaluationType.ENV,  # Minimal evaluation
                seed=run_seed,
                audio_native_config=audio_native_config,
                verbose_logs=False,
            )

            # Consider it successful if simulation completed
            return True, None

        except Exception as e:
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            return False, error_msg

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_single, i): i for i in range(concurrency)}

        for future in as_completed(futures):
            success, error = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                if error:
                    errors.append(error)

    duration = time.time() - start_time

    # Categorize errors
    error_categories = {}
    for err in errors:
        cat = categorize_error(err)
        error_categories[cat] = error_categories.get(cat, 0) + 1

    return ConcurrencyTestResult(
        concurrency=concurrency,
        successful=successful,
        failed=failed,
        errors=errors,
        error_categories=error_categories,
        duration_seconds=duration,
    )


def find_max_concurrency(
    domain: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    task_id: Optional[str] = None,
    max_test_concurrency: int = 64,
    test_duration_seconds: int = 30,
    llm_agent: str = "openai/gpt-4o",
    llm_user: str = "openai/gpt-4o",
    tick_duration: float = 0.2,
    seed: int = 42,
    audio_native: bool = False,
    verbose: bool = False,
    output_path: Optional[Path] = None,
) -> ConcurrencyTestReport:
    """
    Find the maximum concurrency that works without errors.

    Uses exponential search followed by binary search for efficiency.

    Args:
        domain: Domain to test (e.g., "airline", "retail")
        provider: Audio-native provider ("openai" or "gemini")
        model: Specific model to use (or provider default)
        task_id: Specific task ID to use (or picks first task)
        max_test_concurrency: Maximum concurrency to test
        test_duration_seconds: Duration for each test simulation
        llm_agent: LLM model for agent (non-audio-native mode)
        llm_user: LLM model for user simulator
        tick_duration: Tick duration in seconds (audio-native mode)
        seed: Random seed
        audio_native: Whether to use audio-native mode
        verbose: Enable verbose output
        output_path: Path to save JSON results (optional)

    Returns:
        ConcurrencyTestReport with all results and recommendations
    """
    # Configure logging
    logger.remove()
    if verbose:
        logger.add(lambda msg: print(msg), level="DEBUG")
    else:
        logger.add(lambda msg: print(msg), level="WARNING")

    # Load tasks and pick one
    print(f"\n{'=' * 60}")
    print("Concurrency Test Configuration")
    print(f"{'=' * 60}")
    print(f"  Domain: {domain}")
    print(f"  Mode: {'audio-native' if audio_native else 'REST API'}")
    if audio_native:
        effective_model = (
            model if model else DEFAULT_AUDIO_NATIVE_MODELS.get(provider, "unknown")
        )
        print(f"  Provider: {provider}")
        print(f"  Model: {effective_model}")
    else:
        print(f"  Agent LLM: {llm_agent}")
        print(f"  User LLM: {llm_user}")
    print(f"  Test duration: {test_duration_seconds}s per simulation")
    print(f"  Max test concurrency: {max_test_concurrency}")
    print(f"{'=' * 60}\n")

    tasks = get_tasks(task_set_name=domain, num_tasks=10)
    if not tasks:
        raise ValueError(f"No tasks found for domain: {domain}")

    if task_id:
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            raise ValueError(f"Task {task_id} not found in domain {domain}")
    else:
        task = tasks[0]  # Pick first task

    print(f"Using task: {task.id}")

    # Create audio-native config if enabled
    audio_native_config = None
    if audio_native:
        if not provider:
            provider = "openai"

        # Use provider-specific default model if not specified
        effective_model = model if model else DEFAULT_AUDIO_NATIVE_MODELS[provider]

        config_kwargs = {
            "provider": provider,
            "model": effective_model,
            "tick_duration_seconds": tick_duration,
            "max_steps_seconds": test_duration_seconds,
        }

        audio_native_config = AudioNativeConfig(**config_kwargs)

    # Calculate max_steps based on duration and tick
    if audio_native:
        max_steps = int(test_duration_seconds / tick_duration)
    else:
        max_steps = 50  # Reasonable default for REST API mode

    # Phase 1: Exponential search to find approximate limit
    print("\n" + "-" * 40)
    print("Phase 1: Exponential Search")
    print("-" * 40)

    last_successful = 0
    first_failed = None
    current = 1
    results = []

    while current <= max_test_concurrency:
        print(f"\nTesting concurrency: {current}")

        result = test_concurrency_level(
            domain=domain,
            task=task,
            concurrency=current,
            audio_native_config=audio_native_config,
            llm_agent=llm_agent,
            llm_user=llm_user,
            max_steps=max_steps,
            seed=seed,
        )
        results.append(result)

        # Display result
        status = "✅" if result.all_successful else "❌"
        print(
            f"  {status} {result.successful}/{result.concurrency} successful "
            f"({result.duration_seconds:.1f}s)"
        )

        if result.errors:
            print(f"  Errors: {result.error_categories}")
            if verbose:
                for err in result.errors[:3]:  # Show first 3 errors
                    print(f"    - {err[:100]}...")

        if result.all_successful:
            last_successful = current
            current *= 2
        else:
            first_failed = current
            break

    # If we hit max without failures, we're done
    if first_failed is None:
        print(f"\n✅ All tests passed up to max ({max_test_concurrency})")
        effective_model = (
            model if model else DEFAULT_AUDIO_NATIVE_MODELS.get(provider, None)
        )
        report = ConcurrencyTestReport(
            domain=domain,
            mode="audio-native" if audio_native else "rest-api",
            provider=provider if audio_native else None,
            model=effective_model if audio_native else None,
            llm_agent=llm_agent if not audio_native else None,
            llm_user=llm_user,
            task_id=task.id,
            test_duration_seconds=test_duration_seconds,
            max_test_concurrency=max_test_concurrency,
            seed=seed,
            results=results,
            max_successful_concurrency=max_test_concurrency,
            first_failed_concurrency=None,
        )
        if output_path:
            report.save(output_path)
        return report

    # Phase 2: Binary search between last_successful and first_failed
    if first_failed - last_successful > 1:
        print("\n" + "-" * 40)
        print(f"Phase 2: Binary Search ({last_successful} - {first_failed})")
        print("-" * 40)

        low = last_successful
        high = first_failed

        while high - low > 1:
            mid = (low + high) // 2
            print(f"\nTesting concurrency: {mid}")

            result = test_concurrency_level(
                domain=domain,
                task=task,
                concurrency=mid,
                audio_native_config=audio_native_config,
                llm_agent=llm_agent,
                llm_user=llm_user,
                max_steps=max_steps,
                seed=seed,
            )
            results.append(result)

            status = "✅" if result.all_successful else "❌"
            print(
                f"  {status} {result.successful}/{result.concurrency} successful "
                f"({result.duration_seconds:.1f}s)"
            )

            if result.all_successful:
                low = mid
            else:
                high = mid
                if result.errors:
                    print(f"  Errors: {result.error_categories}")

        last_successful = low

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n✅ Maximum working concurrency: {last_successful}")
    print(f"❌ First failing concurrency: {first_failed}")
    print()

    # Recommendations
    print("Recommendations:")
    if last_successful >= 16:
        print(f"  • Use --max-concurrency {last_successful} for full runs")
        print(f"  • Consider {last_successful - 2} for safety margin")
    elif last_successful >= 4:
        print(f"  • Use --max-concurrency {last_successful} for production")
        print("  • Rate limiting may affect longer runs")
    else:
        print("  • Concurrency is heavily limited")
        print("  • Check your API quotas/limits")
        print("  • Consider using a different provider or tier")

    # Build and save report
    effective_model = (
        model if model else DEFAULT_AUDIO_NATIVE_MODELS.get(provider, None)
    )
    report = ConcurrencyTestReport(
        domain=domain,
        mode="audio-native" if audio_native else "rest-api",
        provider=provider if audio_native else None,
        model=effective_model if audio_native else None,
        llm_agent=llm_agent if not audio_native else None,
        llm_user=llm_user,
        task_id=task.id,
        test_duration_seconds=test_duration_seconds,
        max_test_concurrency=max_test_concurrency,
        seed=seed,
        results=results,
        max_successful_concurrency=last_successful,
        first_failed_concurrency=first_failed,
    )

    if output_path:
        report.save(output_path)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Test maximum concurrency for tau2 experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test REST API concurrency (default)
    python scripts/test_concurrency.py --domain airline

    # Test OpenAI Realtime API (audio-native)
    python scripts/test_concurrency.py --domain airline --audio-native --provider openai

    # Test Gemini Live API
    python scripts/test_concurrency.py --domain retail --audio-native --provider gemini

    # Thorough test with longer simulations
    python scripts/test_concurrency.py --domain airline --audio-native --provider openai \\
        --test-duration 60 --max-test 64
        """,
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain to test (e.g., airline, retail)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini"],
        default="openai",
        help="Audio-native provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to use (default: provider's default)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Specific task ID to use for testing",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=64,
        help="Maximum concurrency to test (default: 64)",
    )
    parser.add_argument(
        "--test-duration",
        type=int,
        default=30,
        help="Duration of each test simulation in seconds (default: 30)",
    )
    parser.add_argument(
        "--llm-agent",
        type=str,
        default="openai/gpt-4o",
        help="Agent LLM for non-audio-native mode",
    )
    parser.add_argument(
        "--llm-user",
        type=str,
        default="openai/gpt-4o",
        help="User LLM for simulations",
    )
    parser.add_argument(
        "--tick-duration",
        type=float,
        default=0.2,
        help="Tick duration in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--audio-native",
        action="store_true",
        help="Test audio-native mode (WebSocket) instead of REST API",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save JSON results (default: auto-generated in data/tmp/concurrency_tests/)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Determine output path
    output_path = None
    if not args.no_save:
        if args.output:
            output_path = Path(args.output)
        else:
            # Auto-generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_str = "audio_native" if args.audio_native else "rest_api"
            provider_str = f"_{args.provider}" if args.audio_native else ""
            filename = f"concurrency_test_{args.domain}_{mode_str}{provider_str}_{timestamp}.json"
            output_path = (
                Path(__file__).parent.parent
                / "data"
                / "tmp"
                / "concurrency_tests"
                / filename
            )

    try:
        report = find_max_concurrency(
            domain=args.domain,
            provider=args.provider,
            model=args.model,
            task_id=args.task_id,
            max_test_concurrency=args.max_test,
            test_duration_seconds=args.test_duration,
            llm_agent=args.llm_agent,
            llm_user=args.llm_user,
            tick_duration=args.tick_duration,
            seed=args.seed,
            audio_native=args.audio_native,
            verbose=args.verbose,
            output_path=output_path,
        )

        print(f"\nMax concurrency found: {report.max_successful_concurrency}")
        return 0

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
