#!/usr/bin/env python3
"""
Find the best simulation for a speech activity timeline illustration.

Scores simulations based on the presence of various features that would
make for an informative paper figure.

Usage:
    python scripts/find_best_timeline_simulation.py <results.json> [--top N]

Example:
    python scripts/find_best_timeline_simulation.py data/exp/tau-voice-experiments/experiment_2025_01_22_v4_regular/retail_regular_gemini/results.json --top 10

To generate a timeline with waveform:
    python scripts/find_best_timeline_simulation.py results.json --generate 72 --output timeline.pdf
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class SimulationScore:
    """Score breakdown for a simulation."""

    sim_index: int
    simulation_id: str
    task_id: str
    reward: Optional[float]

    # Counts
    n_user_segments: int = 0
    n_agent_segments: int = 0
    n_user_interruptions: int = 0
    n_backchannels: int = 0
    n_agent_interruptions: int = 0

    # Audio effects (in-segment + out-of-turn)
    n_burst_noise: int = 0
    n_vocal_tics: int = 0
    n_non_directed: int = 0
    n_muffling: int = 0
    n_out_of_turn_effects: int = 0

    # Diagnostic events
    n_no_response: int = 0
    n_no_yield: int = 0

    # Duration
    duration_sec: float = 0.0
    n_ticks: int = 0

    # Total score
    total_score: float = 0.0

    def compute_score(self) -> float:
        """Compute weighted score for timeline illustration suitability."""
        score = 0.0

        # Turns - baseline points
        score += min(self.n_user_segments, 10) * 1.0
        score += min(self.n_agent_segments, 10) * 1.0

        # Interruptions - highly valuable
        score += self.n_user_interruptions * 15.0
        score += self.n_agent_interruptions * 15.0

        # Backchannels - valuable
        score += self.n_backchannels * 12.0

        # Audio effects - variety is good
        if self.n_burst_noise > 0:
            score += 10.0 + min(self.n_burst_noise - 1, 2) * 2.0
        if self.n_vocal_tics > 0:
            score += 10.0 + min(self.n_vocal_tics - 1, 2) * 2.0
        if self.n_non_directed > 0:
            score += 10.0 + min(self.n_non_directed - 1, 2) * 2.0
        if self.n_muffling > 0:
            score += 8.0 + min(self.n_muffling - 1, 2) * 2.0

        # Diagnostic events
        score += self.n_no_response * 5.0
        score += self.n_no_yield * 5.0

        # Duration penalty - prefer 30-90 seconds
        if self.duration_sec < 20:
            score -= (20 - self.duration_sec) * 0.5
        elif self.duration_sec > 120:
            score -= (self.duration_sec - 120) * 0.2

        # Bonus for task success
        if self.reward is not None and self.reward > 0:
            score += 5.0

        self.total_score = score
        return score

    def features_str(self) -> str:
        """Return a string listing all features present."""
        features = []
        if self.n_user_interruptions > 0:
            features.append(f"{self.n_user_interruptions} user_int")
        if self.n_agent_interruptions > 0:
            features.append(f"{self.n_agent_interruptions} agent_int")
        if self.n_backchannels > 0:
            features.append(f"{self.n_backchannels} backchannel")
        if self.n_burst_noise > 0:
            features.append(f"{self.n_burst_noise} burst")
        if self.n_vocal_tics > 0:
            features.append(f"{self.n_vocal_tics} tics")
        if self.n_non_directed > 0:
            features.append(f"{self.n_non_directed} non_dir")
        if self.n_muffling > 0:
            features.append(f"{self.n_muffling} muffle")
        if self.n_no_response > 0:
            features.append(f"{self.n_no_response} no_resp")
        if self.n_no_yield > 0:
            features.append(f"{self.n_no_yield} no_yield")
        return ", ".join(features) if features else "none"


def get_audio_path(
    results_path: Path, task_id: str, simulation_id: str
) -> Optional[Path]:
    """
    Get the path to the both.wav audio file for a simulation.

    Structure: <results_dir>/tasks/task_<task_id>/sim_<simulation_id>/audio/both.wav
    """
    results_dir = results_path.parent
    audio_path = (
        results_dir
        / "tasks"
        / f"task_{task_id}"
        / f"sim_{simulation_id}"
        / "audio"
        / "both.wav"
    )
    if audio_path.exists():
        return audio_path
    return None


def score_simulation(
    sim, sim_index: int, tick_duration: float = 0.2
) -> Optional[SimulationScore]:
    """Score a single simulation for timeline illustration suitability."""
    from experiments.tau_voice.exp.voice_analysis import (
        extract_all_segments,
        extract_interruption_events,
        extract_out_of_turn_effects,
        extract_turn_transitions,
        filter_end_of_conversation_ticks,
    )

    if not sim.ticks:
        return None

    ticks = filter_end_of_conversation_ticks(sim.ticks)
    if not ticks:
        return None

    user_segs, agent_segs = extract_all_segments(ticks, tick_duration)

    reward = sim.reward_info.reward if sim.reward_info else None
    score = SimulationScore(
        sim_index=sim_index,
        simulation_id=sim.id,
        task_id=sim.task_id,
        reward=reward,
        n_ticks=len(ticks),
        duration_sec=len(ticks) * tick_duration,
        n_user_segments=len(user_segs),
        n_agent_segments=len(agent_segs),
    )

    # Count user segment features
    for seg in user_segs:
        if seg.is_interruption:
            score.n_user_interruptions += 1
        if seg.is_backchannel:
            score.n_backchannels += 1
        if seg.has_burst_noise:
            score.n_burst_noise += 1
        if seg.has_vocal_tic:
            score.n_vocal_tics += 1
        if seg.has_non_directed_speech:
            score.n_non_directed += 1
        if seg.has_muffling:
            score.n_muffling += 1

    # Count agent segment features
    for seg in agent_segs:
        if seg.other_speaking_at_start:
            score.n_agent_interruptions += 1

    # Out-of-turn effects - count by type
    out_of_turn = extract_out_of_turn_effects(ticks, tick_duration)
    score.n_out_of_turn_effects = len(out_of_turn)
    for effect in out_of_turn:
        if effect.effect_type == "burst_noise":
            score.n_burst_noise += 1
        elif effect.effect_type == "vocal_tic":
            score.n_vocal_tics += 1
        elif effect.effect_type == "non_directed_speech":
            score.n_non_directed += 1
        elif effect.effect_type == "muffling":
            score.n_muffling += 1

    # Diagnostic events
    try:
        turn_transitions = extract_turn_transitions(
            user_segs, agent_segs, simulation_id=sim.id, task_id=sim.task_id
        )
        score.n_no_response = sum(
            1 for t in turn_transitions if t.outcome == "no_response"
        )
    except Exception:
        pass

    try:
        interruption_events = extract_interruption_events(
            user_segs, agent_segs, ticks, tick_duration
        )
        score.n_no_yield = sum(
            1
            for e in interruption_events
            if e.event_type == "user_interrupts_agent" and not e.interrupted_yielded
        )
    except Exception:
        pass

    score.compute_score()
    return score


def generate_timeline(
    results_path: Path, sim_index: int, output_path: Path, with_audio: bool = True
):
    """Generate a timeline for the specified simulation."""
    from experiments.tau_voice.exp.voice_analysis import (
        extract_all_segments,
        extract_frame_drops,
        extract_interruption_events,
        extract_out_of_turn_effects,
        extract_turn_transitions,
        filter_end_of_conversation_ticks,
        get_tick_duration,
        save_speech_timeline,
    )
    from tau2.data_model.simulation import Results

    print(f"Loading results from: {results_path}")
    results = Results.load(results_path)

    if sim_index >= len(results.simulations):
        print(
            f"Error: sim_index {sim_index} out of range (max: {len(results.simulations) - 1})"
        )
        return

    sim = results.simulations[sim_index]
    print(f"Simulation: {sim.id}")
    print(f"Task ID: {sim.task_id}")

    # Extract domain and agent LLM from results info
    domain = (
        results.info.environment_info.domain_name
        if results.info and results.info.environment_info
        else ""
    )
    agent_llm = ""
    if results.info and results.info.audio_native_config:
        agent_llm = results.info.audio_native_config.model
    elif results.info and results.info.agent_info:
        agent_llm = results.info.agent_info.llm or ""

    # Extract background noise from simulation's speech environment
    background_noise = ""
    if sim.speech_environment and hasattr(
        sim.speech_environment, "background_noise_file"
    ):
        background_noise = sim.speech_environment.background_noise_file or ""

    print(f"Domain: {domain}")
    print(f"Agent LLM: {agent_llm}")
    print(f"Background noise: {background_noise}")

    if not sim.ticks:
        print("Error: No ticks found in simulation")
        return

    tick_duration = get_tick_duration(sim, 0.2)
    print(f"Tick duration: {tick_duration}s")

    ticks = filter_end_of_conversation_ticks(sim.ticks)
    user_segs, agent_segs = extract_all_segments(ticks, tick_duration)
    total_duration = len(ticks) * tick_duration

    print(f"Segments: {len(user_segs)} user, {len(agent_segs)} agent")
    print(f"Duration: {total_duration:.1f}s")

    # Extract diagnostic events
    print("Extracting diagnostic events...")
    turn_transitions = extract_turn_transitions(
        user_segs,
        agent_segs,
        simulation_id=sim.id,
        task_id=sim.task_id,
    )
    out_of_turn = extract_out_of_turn_effects(ticks, tick_duration)
    interruption_events = extract_interruption_events(
        user_segs,
        agent_segs,
        ticks,
        tick_duration,
        out_of_turn_effects=out_of_turn,
    )
    frame_drops = extract_frame_drops(ticks, tick_duration)

    n_no_response = sum(1 for t in turn_transitions if t.outcome == "no_response")
    n_no_yield = sum(
        1
        for e in interruption_events
        if e.event_type == "user_interrupts_agent" and not e.interrupted_yielded
    )
    print(f"  No-response events: {n_no_response}")
    print(f"  No-yield events: {n_no_yield}")
    print(f"  Out-of-turn effects: {len(out_of_turn)}")
    print(f"  Frame drops: {len(frame_drops)}")

    # Get audio path
    audio_path = None
    if with_audio:
        audio_path = get_audio_path(results_path, sim.task_id, sim.id)
        if audio_path:
            print(f"  Audio file: {audio_path}")
        else:
            print("  Audio file: not found")

    # Generate timeline
    save_speech_timeline(
        user_segs,
        agent_segs,
        output_path,
        total_duration_sec=total_duration,
        simulation_id=sim.id,
        task_id=sim.task_id,
        domain=domain,
        agent_llm=agent_llm,
        background_noise=background_noise,
        turn_transitions=turn_transitions,
        interruption_events=interruption_events,
        out_of_turn_effects=out_of_turn,
        frame_drops=frame_drops,
        audio_path=audio_path,
    )
    print(f"\nSaved timeline to: {output_path}")


def main():
    from experiments.tau_voice.exp.voice_analysis import get_tick_duration
    from tau2.data_model.simulation import Results

    parser = argparse.ArgumentParser(
        description="Find the best simulation for timeline illustration"
    )
    parser.add_argument("results_path", type=str, help="Path to results.json file")
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top results to show (default: 20)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=0, help="Minimum duration in seconds"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=float("inf"),
        help="Maximum duration in seconds",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only show successful simulations (reward > 0)",
    )
    parser.add_argument(
        "--generate",
        type=int,
        metavar="INDEX",
        help="Generate timeline for simulation at INDEX",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="timeline.pdf",
        help="Output path for generated timeline (default: timeline.pdf)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Don't include audio waveform in timeline",
    )
    args = parser.parse_args()

    results_path = Path(args.results_path)

    # Generate mode
    if args.generate is not None:
        output_path = Path(args.output)
        generate_timeline(
            results_path, args.generate, output_path, with_audio=not args.no_audio
        )
        return

    # Search mode
    print(f"Loading results from: {results_path}")
    results = Results.load(results_path)
    print(f"Loaded {len(results.simulations)} simulations\n")

    # Score all simulations
    scores: List[SimulationScore] = []
    for i, sim in enumerate(results.simulations):
        if i % 20 == 0:
            print(f"Processing simulation {i}/{len(results.simulations)}...", end="\r")

        tick_duration = get_tick_duration(sim, 0.2)
        score = score_simulation(sim, i, tick_duration)
        if score:
            # Apply filters
            if score.duration_sec < args.min_duration:
                continue
            if score.duration_sec > args.max_duration:
                continue
            if args.success_only and (score.reward is None or score.reward <= 0):
                continue
            scores.append(score)

    print(f"\nScored {len(scores)} simulations with ticks")

    # Sort by score
    scores.sort(key=lambda s: s.total_score, reverse=True)

    # Print top N
    print("\n" + "=" * 120)
    print(f"TOP {args.top} SIMULATIONS FOR TIMELINE ILLUSTRATION")
    print("=" * 120)

    for i, score in enumerate(scores[: args.top]):
        print(
            f"\n{i + 1:2d}. Index {score.sim_index:3d} | Task {score.task_id:4s} | "
            f"Score: {score.total_score:5.1f} | Duration: {score.duration_sec:5.1f}s | "
            f"Segs: {score.n_user_segments:2d}u/{score.n_agent_segments:2d}a | "
            f"Reward: {score.reward}"
        )
        print(f"    Features: [{score.features_str()}]")

    # Print feature distribution
    print("\n" + "=" * 120)
    print("FEATURE DISTRIBUTION")
    print("=" * 120)
    print(
        f"User interruptions: {sum(s.n_user_interruptions for s in scores)} total, {sum(1 for s in scores if s.n_user_interruptions > 0)} sims"
    )
    print(
        f"Agent interruptions: {sum(s.n_agent_interruptions for s in scores)} total, {sum(1 for s in scores if s.n_agent_interruptions > 0)} sims"
    )
    print(
        f"Backchannels: {sum(s.n_backchannels for s in scores)} total, {sum(1 for s in scores if s.n_backchannels > 0)} sims"
    )
    print(
        f"Burst noise: {sum(s.n_burst_noise for s in scores)} total, {sum(1 for s in scores if s.n_burst_noise > 0)} sims"
    )
    print(
        f"Vocal tics: {sum(s.n_vocal_tics for s in scores)} total, {sum(1 for s in scores if s.n_vocal_tics > 0)} sims"
    )
    print(
        f"Non-directed: {sum(s.n_non_directed for s in scores)} total, {sum(1 for s in scores if s.n_non_directed > 0)} sims"
    )
    print(
        f"Muffling: {sum(s.n_muffling for s in scores)} total, {sum(1 for s in scores if s.n_muffling > 0)} sims"
    )

    # Print command to generate timeline
    if scores:
        best = scores[0]
        print("\n" + "=" * 120)
        print("TO GENERATE TIMELINE FOR THE BEST SIMULATION:")
        print("=" * 120)
        print(
            f"\npython scripts/find_best_timeline_simulation.py {results_path} --generate {best.sim_index} --output timeline_task{best.task_id}.pdf\n"
        )


if __name__ == "__main__":
    main()
