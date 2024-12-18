import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Literal, Optional

import polars as pl
import torch
from aces.config import (
    END_OF_RECORD_KEY,
    START_OF_RECORD_KEY,
    DerivedPredicateConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    TemporalWindowBounds,
    ToEventWindowBounds,
)
from loguru import logger


class WindowStatus(Enum):
    UNDETERMINED = 0
    ACTIVE = 1
    SATISFIED = 2
    IMPOSSIBLE = 3


class BoundType(Enum):
    TEMPORAL = "temporal"
    EVENT = "event"


@dataclass
class WindowBound:
    """Base class for window bounds."""

    reference: str  # 'trigger' or 'window_name.start/end'
    inclusive: bool


@dataclass
class TemporalBound(WindowBound):
    """Fixed time offset from reference point."""

    offset: timedelta
    bound_type: BoundType = field(default=BoundType.TEMPORAL, init=False)


@dataclass
class WindowState:
    """State of a window for one sequence in batch."""

    batch_size: int
    start_time: torch.Tensor = field(init=False)  # [batch_size] float tensor, initialized to torch.nan
    end_time: torch.Tensor = field(init=False)  # [batch_size] float tensor, initialized to torch.nan
    in_window: torch.Tensor = field(init=False)  # [batch_size] bool tensor
    predicate_counts: dict[str, torch.Tensor] = field(default_factory=dict)  # str -> [batch_size] int tensor
    status: torch.Tensor = field(init=False)  # [batch_size] int tensor of WindowStatus values
    waiting_for_next_time: torch.Tensor = field(
        init=False
    )  # [batch_size] float tensor, initialized to torch.nan

    def __post_init__(self):
        """Initialize tensors with proper shapes and default values."""
        self.start_time = torch.full((self.batch_size,), float("nan"))
        self.end_time = torch.full((self.batch_size,), float("nan"))
        self.in_window = torch.zeros(self.batch_size, dtype=torch.bool)
        self.status = torch.full((self.batch_size,), WindowStatus.UNDETERMINED.value, dtype=torch.long)
        self.waiting_for_next_time = torch.full((self.batch_size,), float("nan"))

    def reset(self):
        """Reset state for new sequence."""
        self.start_time.fill_(float("nan"))
        self.end_time.fill_(float("nan"))
        self.in_window.fill_(False)
        for counts in self.predicate_counts.values():
            counts.fill_(0)
        self.status.fill_(WindowStatus.UNDETERMINED.value)
        self.waiting_for_next_time.fill_(float("nan"))


T = None | int


@dataclass
class PredicateTensor:
    """
    Manages tokenized predicates and their value constraints.

    Attributes:
        name: Name of the predicate
        tokens: Tensor of vocabulary indices for the predicate (empty for derived predicates)
        value_limits: Tuple of (min_count, max_count) for predicate constraints
        value_inclusions: Tuple of (min_inclusive, max_inclusive) for threshold handling
        children: List of child PredicateTensors for derived predicates
        is_and: Boolean indicating if this is an AND predicate (vs OR)

    Example usage:
        >>> # Create a simple lab predicate
        >>> lab_predicate = PredicateTensor(
        ...     name="high_lab",
        ...     tokens=torch.tensor([6, 7]),  # Lab codes
        ...     value_limits=(2.0, None),    # Value >= 2.0
        ...     value_inclusions=(True, None),  # Inclusive threshold
        ...     children=[],
        ...     is_and=False
        ... )
        >>> state = WindowState(1)

        >>> # Test normal lab value
        >>> lab_predicate.update_counts(state, torch.tensor([6]), torch.tensor([1.5]))  # Lab below threshold
        >>> state.predicate_counts
        {'high_lab': tensor([0])}

        >>> # Test high lab value
        >>> lab_predicate.update_counts(state, torch.tensor([6]), torch.tensor([2.5]))  # Lab above threshold
        >>> state.predicate_counts
        {'high_lab': tensor([1])}

        >>> # Test edge case exactly at threshold
        >>> # Lab at threshold (inclusive)
        >>> lab_predicate.update_counts(state, torch.tensor([7]), torch.tensor([2.0]))
        >>> state.predicate_counts
        {'high_lab': tensor([2])}

    Derived predicates example:
        >>> # Create child predicates
        >>> high_lab = PredicateTensor(
        ...     name="high_lab",
        ...     tokens=torch.tensor([6]),
        ...     value_limits=(2.0, None),
        ...     value_inclusions=(True, None),
        ...     children=[],
        ...     is_and=False
        ... )
        >>> low_lab = PredicateTensor(
        ...     name="low_lab",
        ...     tokens=torch.tensor([6]),
        ...     value_limits=(None, -2.0),
        ...     value_inclusions=(None, False),
        ...     children=[],
        ...     is_and=False
        ... )

        >>> # Create derived OR predicate
        >>> abnormal_lab = PredicateTensor(
        ...     name="abnormal_lab",
        ...     tokens=torch.tensor([]),
        ...     value_limits=(None, None),
        ...     value_inclusions=(None, None),
        ...     children=[high_lab, low_lab],
        ...     is_and=False
        ... )
        >>> state = WindowState(1)

        >>> # Test high value

        >>> abnormal_lab.update_counts(state, torch.tensor([6]), torch.tensor([3.0]))
        >>> state.predicate_counts
        {'high_lab': tensor([1]), 'low_lab': tensor([0])}
        >>> abnormal_lab.get_count(state)
        tensor([1])

        >>> # Test low value
        >>> abnormal_lab.update_counts(state, torch.tensor([6]), torch.tensor([-2.5]))
        >>> state.predicate_counts
        {'high_lab': tensor([1]), 'low_lab': tensor([1])}
        >>> abnormal_lab.get_count(state)  # OR predicate sums the counts
        tensor([2])

    Constraint checking:
        >>> state = WindowState(1)
        >>> lab_predicate = PredicateTensor(
        ...     name="high_lab",
        ...     tokens=torch.tensor([6]),
        ...     value_limits=(2.0, None),
        ...     value_inclusions=(True, None),
        ...     children=[],
        ...     is_and=False
        ... )

        >>> # Test min/max constraints
        >>> lab_predicate.check_constraints(state, min_count=1, max_count=3)  # No events yet
        tensor([False])
        >>> lab_predicate.update_counts(state, torch.tensor([6]), torch.tensor([2.5]))  # Add qualifying event
        >>> lab_predicate.check_constraints(state, min_count=1, max_count=3)  # Now satisfied
        tensor([True])
        >>> lab_predicate.update_counts(state, torch.tensor([6]), torch.tensor([3.0]))  # Add another
        >>> lab_predicate.update_counts(state, torch.tensor([6]), torch.tensor([3.0]))  # And another
        >>> lab_predicate.update_counts(state, torch.tensor([6]), torch.tensor([3.0]))  # One too many
        >>> lab_predicate.check_constraints(state, min_count=1, max_count=3)  # Exceeds max
        tensor([False])

    Impossibility checking:
        >>> state = WindowState(1)
        >>> lab_predicate = PredicateTensor(
        ...     name="high_lab",
        ...     tokens=torch.tensor([6]),
        ...     value_limits=(2.0, None),
        ...     value_inclusions=(True, None),
        ...     children=[],
        ...     is_and=False
        ... )

        >>> # Test max constraint
        >>> lab_predicate.check_impossible(state, max_count=2)  # No events
        tensor([False])
        >>> lab_predicate.update_counts(state, torch.tensor([6]), torch.tensor([3.0]))
        >>> lab_predicate.update_counts(state, torch.tensor([6]), torch.tensor([3.0]))
        >>> lab_predicate.check_impossible(state, max_count=2)  # At limit
        tensor([False])
        >>> lab_predicate.update_counts(state, torch.tensor([6]), torch.tensor([3.0]))
        >>> lab_predicate.check_impossible(state, max_count=2)  # Over limit
        tensor([True])
    """

    name: str
    tokens: torch.Tensor  # [num_tokens] of vocabulary indices
    value_limits: tuple[float | None, float | None]
    value_inclusions: tuple[bool | None, bool | None]
    children: list["PredicateTensor"]
    is_and: bool

    def update_counts(self, state: WindowState, tokens: torch.Tensor, values: torch.Tensor) -> None:
        """
        Update counts for tokens and values in batch.

        Args:
            state: WindowState with batch_size dimension
            tokens: [batch_size] tensor of token indices
            values: [batch_size] tensor of numeric values
        """
        if not self.children:
            # Plain predicate case
            # Initialize predicate count if needed
            if self.name not in state.predicate_counts:
                state.predicate_counts[self.name] = torch.zeros(state.batch_size, dtype=torch.long)

            # Create mask for matching tokens
            token_mask = (tokens.unsqueeze(1) == self.tokens.unsqueeze(0)).any(dim=1)

            if not token_mask.any():
                return

            # Check value thresholds
            min_val, max_val = self.value_limits
            min_incl, max_incl = self.value_inclusions

            should_count = torch.ones_like(tokens, dtype=torch.bool)

            # Check minimum threshold
            if min_val is not None:
                if min_incl:
                    should_count &= values >= min_val
                else:
                    should_count &= values > min_val

            # Check maximum threshold
            if max_val is not None:
                if max_incl:
                    should_count &= values <= max_val
                else:
                    should_count &= values < max_val

            # Update counts where both token matches and value constraints are met
            increment_mask = token_mask & should_count
            state.predicate_counts[self.name] += increment_mask.long()
        else:
            # Derived predicate case - update all children
            for child in self.children:
                child.update_counts(state, tokens, values)

    def get_count(self, state: WindowState) -> torch.Tensor:
        """
        Get total count for this predicate across batch.

        Returns:
            [batch_size] tensor of counts
        """
        if not self.children:
            # Plain predicate - return stored count
            return state.predicate_counts.get(self.name, torch.zeros(state.batch_size, dtype=torch.long))

        # Derived predicate - combine child counts
        child_counts = torch.stack([child.get_count(state) for child in self.children])

        if self.is_and:
            # AND - use minimum count across children
            return (
                torch.min(child_counts, dim=0)[0]
                if child_counts.size(0) > 0
                else torch.zeros(state.batch_size, dtype=torch.long)
            )
        else:
            # OR - use sum of counts
            return torch.sum(child_counts, dim=0)

    def check_constraints(
        self, state: WindowState, min_count: int | None, max_count: int | None
    ) -> torch.Tensor:
        """
        Check if count constraints are satisfied across batch.

        Returns:
            [batch_size] boolean tensor
        """
        counts = self.get_count(state)
        satisfied = torch.ones(state.batch_size, dtype=torch.bool)

        if min_count is not None:
            satisfied &= counts >= min_count

        if max_count is not None:
            satisfied &= counts <= max_count

        return satisfied

    def check_impossible(self, state: WindowState, max_count: int | None) -> torch.Tensor:
        """
        Check if constraints are impossible to satisfy across batch.

        Returns:
            [batch_size] boolean tensor
        """
        if max_count is None:
            return torch.zeros(state.batch_size, dtype=torch.bool)

        counts = self.get_count(state)
        return counts > max_count

    @classmethod
    def from_config(
        cls, metadata_df: pl.DataFrame, config: TaskExtractorConfig, predicate_name: str
    ) -> "PredicateTensor":
        """Create a PredicateTensor from a task configuration."""
        predicate = config.predicates[predicate_name]

        if isinstance(predicate, PlainPredicateConfig):
            # Handle plain predicate - has tokens but no children
            tokens = metadata_df.filter(predicate.MEDS_eval_expr())["code/vocab_index"].to_torch()

            if tokens.shape[0] == 0:
                logger.warning(f"Predicate {predicate_name} matched no codes")

            value_limits = (predicate.value_min, predicate.value_max)
            value_inclusions = (predicate.value_min_inclusive, predicate.value_max_inclusive)

            return cls(
                name=predicate_name,
                tokens=tokens,
                value_limits=value_limits,
                value_inclusions=value_inclusions,
                children=[],
                is_and=False,
            )

        elif isinstance(predicate, DerivedPredicateConfig):
            # Handle derived predicate - has children but no tokens
            children = [
                cls.from_config(metadata_df, config, child_name) for child_name in predicate.input_predicates
            ]

            return cls(
                name=predicate_name,
                tokens=torch.tensor([], dtype=torch.long),
                value_limits=(None, None),
                value_inclusions=(None, None),
                children=children,
                is_and=predicate.is_and,
            )
        else:
            raise ValueError(f"Unknown predicate type: {type(predicate)}")


def get_predicate_tensor(
    metadata_df: pl.DataFrame, config: TaskExtractorConfig, predicate_name: str
) -> PredicateTensor:
    """
    Create a PredicateTensor from task configuration predicate.

    Args:
        metadata_df: DataFrame containing code/vocab_index mapping
        config: Task configuration
        predicate_name: Name of predicate to create tensor for

    Returns:
        PredicateTensor object containing tokens and value constraints

    Raises:
        ValueError: If predicate type is unknown
    """
    predicate = config.predicates[predicate_name]

    if isinstance(predicate, PlainPredicateConfig):
        # Handle plain predicate
        predicate_tensor = metadata_df.filter(predicate.MEDS_eval_expr())["code/vocab_index"].to_torch()

        if predicate_tensor.shape[0] == 0:
            logger.warning(f"Predicate {predicate_name} returned no rows. Skipping it.")

        value_limits = (predicate.value_min, predicate.value_max)
        value_inclusions = (predicate.value_min_inclusive, predicate.value_max_inclusive)

        return PredicateTensor(
            name=predicate_name,
            tokens=predicate_tensor,
            value_limits=value_limits,
            value_inclusions=value_inclusions,
            children=[],
            is_and=False,
        )

    elif isinstance(predicate, DerivedPredicateConfig):
        # Handle derived (OR/AND) predicate
        child_predicates = []
        value_limits = (None, None)
        value_inclusions = (None, None)

        for child_predicate_name in predicate.input_predicates:
            # Create child PredicateTensor
            child = get_predicate_tensor(metadata_df, config, child_predicate_name)
            child_predicates.append(child)

        return PredicateTensor(
            name=predicate_name,
            tokens=None,
            value_limits=value_limits,
            value_inclusions=value_inclusions,
            children=child_predicates,
            is_and=predicate.is_and,
        )

    else:
        raise ValueError(f"Unknown predicate type {type(predicate)}")


@dataclass
class EventBound(WindowBound):
    """Bound defined by occurrence of events."""

    predicate: PredicateTensor | str
    direction: Literal["next", "previous"]
    bound_type: BoundType = field(default=BoundType.EVENT, init=False)


@dataclass
class WindowNode:
    """Node in autoregressive window tree."""

    name: str
    start_bound: TemporalBound | EventBound
    end_bound: TemporalBound | EventBound
    predicate_constraints: dict[str, tuple[int | None, int | None]]
    label: str | None
    index_timestamp: str | None
    tensorized_predicates: dict[str, PredicateTensor]
    parent: Optional["WindowNode"] = None
    children: list["WindowNode"] = field(default_factory=list)
    state: WindowState | None = None
    ignore: bool = False
    label_value: bool | None = None

    def get_labels(self):
        if self.label is not None:
            if self.label_value is not None:
                return self.label_value
            else:
                return torch.zeros((self.state.batch_size), dtype=torch.bool)
        else:
            for node in self.children:
                label = node.get_labels()
                if label is not None:
                    return label
            return None

    def ignore_windows(self, window_names: list[str]):
        if self.name in window_names:
            self.ignore = True
            self.state.status.fill_(WindowStatus.SATISFIED.value)
        for child in self.children:
            child.ignore_windows(window_names)

    def initialize_batch(self, batch_size: int):
        """Initialize batch states."""
        self.state = WindowState(batch_size=batch_size)
        for child in self.children:
            child.initialize_batch(batch_size)

    def _check_label(self) -> None:
        if self.label is not None:
            label_counts = self._get_count(self.label)
            # TODO: Consider if we need to consider self.state.status impossible cases when updating labels
            self.label_value = label_counts > 0

    def _check_start_condition(self, time_delta: torch.Tensor, event_token: torch.Tensor) -> torch.Tensor:
        """Check if window should start at current time/event."""
        if self.start_bound.bound_type == BoundType.TEMPORAL:
            ref_time = torch.zeros_like(time_delta)  # Default to trigger time
            offset_days = self.start_bound.offset.total_seconds() / (24 * 3600)
            target_time = ref_time + offset_days

            if self.start_bound.inclusive:
                return time_delta >= target_time
            return time_delta > target_time
        else:  # EventBound
            if self.start_bound.direction == "next":
                is_target = event_token == int(self.start_bound.predicate)
                at_or_after_ref = time_delta >= 0
                return is_target & at_or_after_ref
            else:
                raise NotImplementedError("Previous event bounds not yet supported")

    def _check_end_condition(self, time_delta: torch.Tensor, event_token: torch.Tensor) -> torch.Tensor:
        """Check if window should end at current time/event."""
        if self.end_bound.bound_type == BoundType.TEMPORAL:
            ref_time = torch.zeros_like(time_delta)
            offset_days = self.end_bound.offset.total_seconds() / (24 * 3600)
            target_time = ref_time + offset_days

            if self.end_bound.inclusive:
                return time_delta >= target_time
            return time_delta > target_time
        else:  # EventBound
            return False

    def _update_counts(self, event_token: torch.Tensor, numeric_value: torch.Tensor):
        """Update predicate counts for the window."""
        for tensorized_predicate in self.tensorized_predicates.values():
            tensorized_predicate.update_counts(self.state, event_token, numeric_value)

    def _get_count(self, predicate_id: str) -> torch.Tensor:
        predicate_tensor = self.tensorized_predicates[predicate_id]
        return predicate_tensor.get_count(self.state)

    def _check_constraints_satisfied(self) -> torch.Tensor:
        """Check if all predicate constraints are satisfied."""
        satisfied = torch.ones(self.state.batch_size, dtype=torch.bool)

        for pred, (min_count, max_count) in self.predicate_constraints.items():
            count = self._get_count(pred)

            if min_count is not None:
                satisfied &= count >= min_count
            if max_count is not None:
                satisfied &= count <= max_count

        # For trigger window, constraints are satisfied as soon as met
        if self.name == "trigger":
            return satisfied

        # For other windows, need to wait for window end
        if isinstance(self.end_bound, TemporalBound):
            return satisfied & ~torch.isnan(self.state.end_time)
        else:
            if isinstance(self.end_bound.predicate, str):
                raise NotImplementedError("Non-Predicate window End Event Bound predicates not yet supported")
            return satisfied & self.end_bound.predicate.check_constraints(self.state, 1, None)

    def _check_constraints_impossible(self) -> torch.Tensor:
        """Check if constraints are impossible to satisfy."""
        impossible = torch.zeros(self.state.batch_size, dtype=torch.bool)

        for pred, (min_count, max_count) in self.predicate_constraints.items():
            count = self._get_count(pred)

            if max_count is not None:
                impossible |= count > max_count

            if min_count is not None:
                end_time_set = ~torch.isnan(self.state.end_time)
                impossible |= end_time_set & (count < min_count)

        return impossible

    def update(
        self,
        time_deltas: torch.Tensor,
        event_tokens: torch.Tensor,
        numeric_values: torch.Tensor,
    ) -> torch.Tensor:
        """Update state for batch."""
        if self.ignore:
            return torch.full_like(self.state.status, WindowStatus.SATISFIED.value)

        # Handle waiting for next time point
        waiting_mask = ~torch.isnan(self.state.waiting_for_next_time)
        if waiting_mask.any():
            past_wait_time = time_deltas > self.state.waiting_for_next_time
            newly_satisfied = waiting_mask & past_wait_time
            still_waiting = waiting_mask & ~past_wait_time

            self.state.status = torch.where(
                newly_satisfied, torch.tensor(WindowStatus.SATISFIED.value), self.state.status
            )
            if newly_satisfied.any():
                self._check_label()

            # Update counts for those still waiting
            if still_waiting.any():
                self._update_counts(event_tokens, numeric_values)

            # Clear waiting times for satisfied sequences
            self.state.waiting_for_next_time = torch.where(
                newly_satisfied, torch.tensor(float("nan")), self.state.waiting_for_next_time
            )

        # Don't update if already determined
        active_mask = (self.state.status != WindowStatus.SATISFIED.value) & (
            self.state.status != WindowStatus.IMPOSSIBLE.value
        )

        if not active_mask.any():
            return self.state.status

        # Check window start
        start_mask = active_mask & torch.isnan(self.state.start_time)
        should_start = self._check_start_condition(time_deltas, event_tokens)
        new_starts = start_mask & should_start

        if new_starts.any():
            self.state.start_time = torch.where(new_starts, time_deltas, self.state.start_time)
            self.state.in_window |= new_starts
            self.state.status = torch.where(
                new_starts, torch.tensor(WindowStatus.ACTIVE.value), self.state.status
            )

        # Update counts for active windows
        update_mask = self.state.in_window & active_mask
        if update_mask.any():
            self._update_counts(event_tokens, numeric_values)
            self.state.status = torch.where(
                update_mask, torch.tensor(WindowStatus.ACTIVE.value), self.state.status
            )
            self._check_label()

            # Check constraints
            impossible = self._check_constraints_impossible()
            satisfied = self._check_constraints_satisfied()
            if self.label_value is not None:
                satisfied = satisfied | self.label_value

            # Update status based on constraints
            self.state.status = torch.where(
                impossible & update_mask, torch.tensor(WindowStatus.IMPOSSIBLE.value), self.state.status
            )
            self.state.in_window &= ~(impossible & update_mask)

            # Handle satisfied constraints
            newly_satisfied = satisfied & update_mask
            if isinstance(self.end_bound, EventBound) and self.end_bound.inclusive:
                self.state.waiting_for_next_time = torch.where(
                    newly_satisfied, time_deltas, self.state.waiting_for_next_time
                )
            else:
                self.state.status = torch.where(
                    newly_satisfied, torch.tensor(WindowStatus.SATISFIED.value), self.state.status
                )
                self.state.in_window &= ~newly_satisfied

        # Check window end
        should_end = self._check_end_condition(time_deltas, event_tokens)
        end_mask = self.state.in_window & should_end

        if end_mask.any():
            self.state.end_time = torch.where(end_mask, time_deltas, self.state.end_time)
            self.state.in_window &= ~end_mask

            satisfied_at_end = self._check_constraints_satisfied()
            self.state.status = torch.where(
                end_mask & satisfied_at_end,
                torch.tensor(WindowStatus.SATISFIED.value),
                torch.where(end_mask, torch.tensor(WindowStatus.IMPOSSIBLE.value), self.state.status),
            )

        return self.state.status


def process_node(
    node: WindowNode,
    tokens,
    time_deltas,
    numeric_values,
) -> torch.Tensor:  # [batch_size] of status values
    # Update this node's state
    status = node.update(
        time_deltas,
        tokens,
        numeric_values,
    )

    # If this node is satisfied, process children
    satisfied_mask = status == WindowStatus.SATISFIED.value
    if satisfied_mask.any() and node.children:
        for child in node.children:
            child_status = process_node(child, tokens, time_deltas, numeric_values)
            # Update parent status where children are not satisfied
            status = torch.where(
                satisfied_mask & (child_status != WindowStatus.SATISFIED.value), child_status, status
            )

    return status


class AutoregressiveWindowTree:
    """Manages window tree for autoregressive constraint tracking."""

    def __init__(self, root: WindowNode, batch_size: int):
        self.root = root
        self.batch_size = batch_size
        # Initialize states for all nodes
        self.root.initialize_batch(batch_size)

    def update(
        self,
        tokens: torch.Tensor,
        time_deltas: torch.Tensor,
        numeric_values: torch.Tensor,
    ) -> torch.Tensor:  # [batch_size] of ConstraintStatus
        """Process new tokens through tree."""
        return process_node(self.root, tokens, time_deltas, numeric_values)


def calculate_index_timestamp_info(tree: AutoregressiveWindowTree) -> tuple[float, list[str], str]:
    """Calculate the temporal gap and identify windows prior to the index timestamp.

    The function traverses the tree to find the node with an index_timestamp,
    calculates the temporal offset to the trigger, and identifies all windows
    that must be processed before reaching the index timestamp.

    Args:
        tree: AutoregressiveWindowTree containing the window nodes

    Returns:
        TimestampInfo containing:
            - gap_days: temporal gap in days between index timestamp and trigger
            - prior_windows: list of window names that come before the index timestamp
            - index_window: name of the window containing the index timestamp

    Raises:
        ValueError: If no index timestamp is found or if multiple index timestamps exist
    """

    def find_index_timestamp_node(node: WindowNode) -> tuple[WindowNode, str] | None:
        """Recursively find the node with an index_timestamp.
        Returns tuple of (node, timestamp_type) where timestamp_type is 'start' or 'end'."""
        # Check current node
        if node.index_timestamp is not None:
            if node.index_timestamp not in ["start", "end"]:
                raise ValueError(f"Invalid index_timestamp value: {node.index_timestamp}")
            return (node, node.index_timestamp)

        # Check children recursively
        for child in node.children:
            result = find_index_timestamp_node(child)
            if result is not None:
                return result

        return None

    # Find node with index timestamp
    result = find_index_timestamp_node(tree.root)
    if result is None:
        raise ValueError("No index timestamp found in tree")

    index_node, timestamp_type = result
    total_offset_days = 0.0
    prior_windows = []

    # Follow path back to trigger, accumulating temporal offsets and windows
    current_node = index_node
    while current_node is not None and current_node.name != "trigger":
        # Add current window to prior windows if it's not the index window
        # or if we're looking at its start time and the index is at its end
        if current_node != index_node or (current_node == index_node and timestamp_type == "end"):
            prior_windows.append(current_node.name)

        bound = current_node.start_bound if timestamp_type == "start" else current_node.end_bound

        # Verify it's a temporal bound
        if bound.bound_type != BoundType.TEMPORAL:
            raise ValueError(f"Non-temporal bound found in path to trigger for node {current_node.name}")

        # Add the offset in days
        offset_days = bound.offset.total_seconds() / (24 * 3600)
        total_offset_days += offset_days

        # Move to parent
        current_node = current_node.parent

        # If moving to parent, we're now looking at the start bound
        timestamp_type = "start"

    # Reverse the list since we collected windows from index back to trigger
    prior_windows.reverse()

    return total_offset_days, prior_windows, index_node.name


def convert_task_config(
    config: TaskExtractorConfig, batch_size: int, metadata_df: pl.DataFrame
) -> AutoregressiveWindowTree:
    """Convert TaskExtractorConfig to AutoregressiveWindowTree.

    Args:
        config: Task configuration from ACES
        batch_size: Size of batches for constraint tracking

    Returns:
        AutoregressiveWindowTree configured according to task config
    """
    # 0. Precache tensorized predicates
    tensorized_predicates = {}
    for p in config.predicates:
        pred_tensor = get_predicate_tensor(metadata_df, config, p)
        tensorized_predicates[p] = pred_tensor

    # 1. Create trigger/root node
    root = WindowNode(
        name="trigger",
        start_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        end_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        predicate_constraints={config.trigger.predicate: (1, 1)},
        label=None,
        index_timestamp=None,
        tensorized_predicates=tensorized_predicates,
    )

    def convert_endpoint_expr(
        config: TaskExtractorConfig,
        metadata_df: pl.DataFrame,
        expr: ToEventWindowBounds | TemporalWindowBounds | None,
        window_name: str,
    ) -> tuple[TemporalBound | EventBound, str]:
        """Convert ACES endpoint expression to our bound type."""
        if expr is None:
            return None, None

        if isinstance(expr, TemporalWindowBounds):
            return (
                TemporalBound(reference=window_name, inclusive=expr.left_inclusive, offset=expr.window_size),
                window_name,
            )
        else:  # ToEventWindowBounds
            direction = "previous" if expr.end_event.startswith("-") else "next"
            predicate = expr.end_event.lstrip("-")
            if predicate not in [END_OF_RECORD_KEY, START_OF_RECORD_KEY]:
                predicate = get_predicate_tensor(metadata_df, config, predicate)

            return (
                EventBound(
                    reference=window_name,
                    inclusive=expr.right_inclusive,
                    predicate=predicate,
                    direction=direction,
                ),
                window_name,
            )

    # 2. Process each window definition and create nodes
    all_nodes = {"trigger": root}
    for window_name, window in config.windows.items():
        logger.info(f"Processing window {window_name}")
        # Convert start/end expressions
        start_bound, start_ref = convert_endpoint_expr(
            config, metadata_df, window.start_endpoint_expr, f"{window_name}.start"
        )
        end_bound, end_ref = convert_endpoint_expr(
            config, metadata_df, window.end_endpoint_expr, f"{window_name}.end"
        )

        # Create window node with converted bounds
        if start_bound is None:
            parent_window_name, start_or_end = window.start.split(".")
            if start_or_end == "start":
                start_bound = all_nodes[parent_window_name].start_bound
            elif start_or_end == "end":
                start_bound = all_nodes[parent_window_name].end_bound
        if end_bound is None:
            if window.end == "trigger":
                end_bound = root.end_bound
            else:
                parent_window_name, start_or_end = window.end.split(".")
                if start_or_end == "start":
                    end_bound = all_nodes[parent_window_name].start_bound
                elif start_or_end == "end":
                    end_bound = all_nodes[parent_window_name].end_bound
        node = WindowNode(
            name=window_name,
            start_bound=start_bound,
            end_bound=end_bound,
            predicate_constraints=window.has,
            label=window.label,
            index_timestamp=window.index_timestamp,
            tensorized_predicates=tensorized_predicates,
        )
        all_nodes[window_name] = node

        # Set up parent relationship
        if window.referenced_event[0] == "trigger":
            root.children.append(node)
            node.parent = root
        else:
            parent_window = window.referenced_event[0]
            parent_node = all_nodes[parent_window]
            parent_node.children.append(node)
            node.parent = parent_node

    # Create tree from root
    tree = AutoregressiveWindowTree(root, batch_size)
    return tree


TimeScale = Literal["Y", "M", "W", "D", "h", "m", "s"]


@dataclass
class SequenceLabeler:
    tree: AutoregressiveWindowTree
    gap_days: float
    batch_size: int
    time_scale: TimeScale = "D"  # Default to days
    _finished: bool = False

    def _convert_to_days(self, times: torch.Tensor) -> torch.Tensor:
        """Convert times from the specified time scale to days."""
        if self.time_scale == "Y":
            return times * 365  # Approximate
        elif self.time_scale == "M":
            return times * 30  # Approximate
        elif self.time_scale == "W":
            return times * 7
        elif self.time_scale == "D":
            return times
        elif self.time_scale == "h":
            return times / 24
        elif self.time_scale == "m":
            return times / (24 * 60)
        elif self.time_scale == "s":
            return times / (24 * 60 * 60)
        else:
            raise ValueError(f"Unknown time scale: {self.time_scale}")

    @classmethod
    def from_yaml_str(
        cls, yaml_str: str, metadata_df: pl.DataFrame, batch_size: int, time_scale: TimeScale = "D"
    ) -> "SequenceLabeler":
        """Create a labeler from YAML task configuration string."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
            f.write(yaml_str)
            f.flush()
            task_config = TaskExtractorConfig.load(f.name)
        return cls.from_task_config(task_config, metadata_df, batch_size, time_scale)

    @classmethod
    def from_yaml_file(
        cls, yaml_path: str, metadata_df: pl.DataFrame, batch_size: int, time_scale: TimeScale = "D"
    ) -> "SequenceLabeler":
        """Create a labeler from YAML task configuration file."""
        task_config = TaskExtractorConfig.load(yaml_path)
        return cls.from_task_config(task_config, metadata_df, batch_size, time_scale)

    @classmethod
    def from_task_config(
        cls,
        config: TaskExtractorConfig,
        metadata_df: pl.DataFrame,
        batch_size: int,
        time_scale: TimeScale = "D",
    ) -> "SequenceLabeler":
        """Create a labeler from TaskExtractorConfig object."""
        tree = convert_task_config(config, batch_size, metadata_df)
        gap_days, prior_windows, _ = calculate_index_timestamp_info(tree)

        # Initialize tree
        tree.root.initialize_batch(batch_size)
        tree.root.ignore_windows(prior_windows + ["trigger"])

        return cls(tree=tree, gap_days=gap_days, batch_size=batch_size, time_scale=time_scale)

    def process_step(
        self,
        tokens: torch.Tensor | list[int],
        times: torch.Tensor | list[float],
        values: torch.Tensor | list[float],
    ) -> torch.Tensor:
        """
        Process one step of token sequences.

        Args:
            tokens: Token IDs for current step [batch_size]
            times: Time values for current step [batch_size] in specified time_scale units
            values: Numeric values for current step [batch_size]

        Returns:
            Tensor of status values for each sequence [batch_size]
        """
        # Convert inputs to tensors if needed
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        if not isinstance(times, torch.Tensor):
            times = torch.tensor(times, dtype=torch.float)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float)

        # Ensure proper shapes
        tokens = tokens.view(self.batch_size)
        times = times.view(self.batch_size)
        values = values.view(self.batch_size)

        # Convert times to days and adjust by gap days
        times_in_days = self._convert_to_days(times)
        time_deltas = times_in_days + self.gap_days

        # Update tree state
        status = self.tree.update(tokens, time_deltas, values)

        # Check if we're done
        self._finished = ((status == 2) | (status == 3)).all()

        return status

    def get_labels(self) -> torch.Tensor:
        """Get final binary labels for each sequence in the batch."""
        return self.tree.root.get_labels()

    def is_finished(self) -> bool:
        """Check if all sequences have reached a final state."""
        return self._finished
