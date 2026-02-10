"""
tau2.run_new -- Clean entry point for running simulations.

This module is the intended replacement for tau2.run. It is a thin facade
that delegates to the tau2.runner package and re-exports commonly used
symbols so callers can do:

    from tau2.run_new import run_domain, get_tasks, EvaluationType

The old run.py had ~1300 lines mixing batch orchestration, environment
construction, voice setup, checkpointing, audio saving, review, and
logging all in one file. This module delegates all of that to the runner
package's layered architecture:

    Layer 1 (simulation.py):  run_simulation()
    Layer 2 (build.py):       build_* functions
    Layer 3 (batch.py):       run_domain, run_tasks, run_single_task
    Helpers (helpers.py):     get_tasks, get_options, etc.

Usage:
    # High-level: run all tasks in a domain
    from tau2.run_new import run_domain
    from tau2.data_model.simulation import TextRunConfig
    results = run_domain(TextRunConfig(domain="retail", agent="llm_agent", ...))

    # Mid-level: run a single task
    from tau2.run_new import get_tasks, run_single_task
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    result = run_single_task(config, tasks[0], seed=42)

    # Low-level: build and run manually
    from tau2.run_new import build_orchestrator, run_simulation
    orch = build_orchestrator(config, task, seed=42)
    sim_run = run_simulation(orch)

Note on run_task: The old run.py exposed run_task() with a flat argument
list (domain, task, agent, user, llm_agent, ...). The new runner uses
run_single_task(config, task, ...) which takes a RunConfig instead.
Callers using the old run_task signature should migrate to:
    run_single_task(TextRunConfig(...), task, seed=...)
For now, run_task is NOT aliased here because the signatures are
incompatible. Import from tau2.run if you still need the old API.
"""

# Re-exports for convenience (callers previously got these from tau2.run)
from tau2.data_model.simulation import RunConfig, TextRunConfig, VoiceRunConfig
from tau2.evaluator.evaluator import EvaluationType

# Layer 3: Batch (full evaluation with concurrency, checkpointing, retries)
from tau2.runner.batch import run_domain, run_single_task, run_tasks

# Layer 2: Build (construct instances from config/names)
from tau2.runner.build import (
    build_agent,
    build_environment,
    build_orchestrator,
    build_text_orchestrator,
    build_user,
    build_voice_orchestrator,
    build_voice_user,
)

# Helpers (task loading, metadata, options)
from tau2.runner.helpers import (
    get_environment_info,
    get_info,
    get_options,
    get_tasks,
    load_task_splits,
    load_tasks,
    make_run_name,
)

# Layer 1: Execute (lowest level, no registry dependency)
from tau2.runner.simulation import run_simulation

__all__ = [
    # Layer 1: Execute
    "run_simulation",
    # Layer 2: Build
    "build_environment",
    "build_agent",
    "build_user",
    "build_voice_user",
    "build_orchestrator",
    "build_text_orchestrator",
    "build_voice_orchestrator",
    # Layer 3: Batch
    "run_domain",
    "run_tasks",
    "run_single_task",
    # Helpers
    "get_options",
    "get_environment_info",
    "load_task_splits",
    "load_tasks",
    "get_tasks",
    "make_run_name",
    "get_info",
    # Re-exports
    "RunConfig",
    "TextRunConfig",
    "VoiceRunConfig",
    "EvaluationType",
]
