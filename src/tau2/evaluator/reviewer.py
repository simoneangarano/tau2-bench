"""
Conversation reviewer for analyzing simulation trajectories for errors.

This module provides functionality to review a single simulation using an
LLM judge to identify errors made by the agent and/or user simulator.

Two review modes are supported:
- "full": Review both agent and user simulator errors (also does auth classification)
- "user": Review only user simulator errors

This is different from the evaluator which computes task success rewards/metrics.
The reviewer identifies qualitative conversation errors.

Usage:
    from tau2.evaluator.reviewer import review_simulation, ReviewMode

    # Full review (agent + user errors)
    review, auth = review_simulation(simulation, task, ReviewMode.FULL, ...)
    simulation.review = review
    simulation.auth_classification = auth

    # User-only review
    review, _ = review_simulation(simulation, task, ReviewMode.USER, ...)
    simulation.user_only_review = review
"""

from enum import Enum
from typing import Optional, Union

from tau2.data_model.simulation import (
    AuthenticationClassification,
    Review,
    SimulationRun,
    UserInfo,
    UserOnlyReview,
)
from tau2.data_model.tasks import Task
from tau2.evaluator.auth_classifier import (
    AuthenticationClassifier,
    FullDuplexAuthenticationClassifier,
)
from tau2.evaluator.review_llm_judge import (
    ConversationReviewer,
    FullDuplexConversationReviewer,
)
from tau2.evaluator.review_llm_judge_user_only import (
    FullDuplexUserOnlyReviewer,
    UserOnlyReviewer,
)


class ReviewMode(str, Enum):
    """Review mode."""

    FULL = "full"  # Review both agent and user errors
    USER = "user"  # Review only user simulator errors


def _is_full_duplex(simulation: SimulationRun) -> bool:
    """Check if the simulation used full-duplex mode (has ticks)."""
    return simulation.ticks is not None and len(simulation.ticks) > 0


def review_simulation(
    simulation: SimulationRun,
    task: Task,
    mode: ReviewMode,
    user_info: UserInfo,
    policy: Optional[str] = None,
    interruption_enabled: bool = False,
) -> tuple[Union[Review, UserOnlyReview], Optional[AuthenticationClassification]]:
    """
    Review a single simulation for conversation errors.

    Args:
        simulation: The simulation run to review.
        task: The task specification.
        mode: Review mode - FULL (agent+user) or USER (user only).
        user_info: User info containing simulation guidelines.
        policy: The policy the agent must follow (required for FULL mode).
        interruption_enabled: Whether interruption was enabled (for full-duplex).

    Returns:
        Tuple of (review_result, auth_classification).
        - For FULL mode: (Review, AuthenticationClassification)
        - For USER mode: (UserOnlyReview, None)
    """
    is_full_duplex = _is_full_duplex(simulation)

    if mode == ReviewMode.FULL:
        if not policy:
            raise ValueError("policy is required for FULL review mode")
        # Full review: agent + user errors + auth classification
        if is_full_duplex:
            review = FullDuplexConversationReviewer.review(
                user_info=user_info,
                task=task,
                full_trajectory=simulation.ticks,
                policy=policy,
                interruption_enabled=interruption_enabled,
            )
            auth_classification = FullDuplexAuthenticationClassifier.classify(
                ticks=simulation.ticks,
            )
        else:
            review = ConversationReviewer.review(
                user_info=user_info,
                task=task,
                full_trajectory=simulation.messages,
                policy=policy,
            )
            auth_classification = AuthenticationClassifier.classify(
                messages=simulation.messages,
            )
        return review, auth_classification

    else:  # ReviewMode.USER
        # User-only review: user simulator errors only
        if is_full_duplex:
            review = FullDuplexUserOnlyReviewer.review(
                user_info=user_info,
                task=task,
                full_trajectory=simulation.ticks,
                interruption_enabled=interruption_enabled,
            )
        else:
            review = UserOnlyReviewer.review(
                user_info=user_info,
                task=task,
                full_trajectory=simulation.messages,
            )
        return review, None
