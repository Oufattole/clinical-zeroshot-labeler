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

    start_time: float | None = None
    end_time: float | None = None
    in_window: bool = False
    predicate_counts: dict[str, int] = field(default_factory=dict)
    status: WindowStatus = WindowStatus.UNDETERMINED
    waiting_for_next_time: float | None = None  # Used for end_inclusive event bounds, points to current time

    def reset(self):
        """Reset state for new sequence."""
        self.start_time = None
        self.end_time = None
        self.in_window = False
        self.predicate_counts.clear()
        self.status = WindowStatus.UNDETERMINED
        self.waiting_for_next_time = None


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
        >>> state = WindowState()

        >>> # Test normal lab value
        >>> lab_predicate.update_counts(state, 6, 1.5)  # Lab below threshold
        >>> state.predicate_counts
        {'high_lab': 0}

        >>> # Test high lab value
        >>> lab_predicate.update_counts(state, 6, 2.5)  # Lab above threshold
        >>> state.predicate_counts
        {'high_lab': 1}

        >>> # Test edge case exactly at threshold
        >>> lab_predicate.update_counts(state, 7, 2.0)  # Lab at threshold (inclusive)
        >>> state.predicate_counts
        {'high_lab': 2}

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
        >>> state = WindowState()

        >>> # Test high value
        >>> abnormal_lab.update_counts(state, 6, 3.0)
        >>> state.predicate_counts
        {'high_lab': 1, 'low_lab': 0}
        >>> abnormal_lab.get_count(state)
        1

        >>> # Test low value
        >>> abnormal_lab.update_counts(state, 6, -2.5)
        >>> state.predicate_counts
        {'high_lab': 1, 'low_lab': 1}
        >>> abnormal_lab.get_count(state)  # OR predicate sums the counts
        2

    Constraint checking:
        >>> state = WindowState()
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
        False
        >>> lab_predicate.update_counts(state, 6, 2.5)  # Add qualifying event
        >>> lab_predicate.check_constraints(state, min_count=1, max_count=3)  # Now satisfied
        True
        >>> lab_predicate.update_counts(state, 6, 3.0)  # Add another
        >>> lab_predicate.update_counts(state, 6, 3.0)  # And another
        >>> lab_predicate.update_counts(state, 6, 3.0)  # One too many
        >>> lab_predicate.check_constraints(state, min_count=1, max_count=3)  # Exceeds max
        False

    Impossibility checking:
        >>> state = WindowState()
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
        False
        >>> lab_predicate.update_counts(state, 6, 3.0)
        >>> lab_predicate.update_counts(state, 6, 3.0)
        >>> lab_predicate.check_impossible(state, max_count=2)  # At limit
        False
        >>> lab_predicate.update_counts(state, 6, 3.0)
        >>> lab_predicate.check_impossible(state, max_count=2)  # Over limit
        True
    """

    name: str
    tokens: torch.Tensor
    value_limits: tuple[float | None, float | None]
    value_inclusions: tuple[bool | None, bool | None]
    children: list["PredicateTensor"]
    is_and: bool

    @classmethod
    def from_config(
        cls, metadata_df: pl.DataFrame, config: TaskExtractorConfig, predicate_name: str
    ) -> "PredicateTensor":
        """
        Create a PredicateTensor from a task configuration.
        """
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
            children = []

            # Process each child predicate
            for child_name in predicate.input_predicates:
                child = cls.from_config(metadata_df, config, child_name)
                children.append(child)

            value_limits = (predicate.value_min, predicate.value_max)
            value_inclusions = (predicate.value_min_inclusive, predicate.value_max_inclusive)

            return cls(
                name=predicate_name,
                tokens=torch.tensor([], dtype=torch.long),  # Empty for derived predicates
                value_limits=value_limits,  # Empty for derived predicates
                value_inclusions=value_inclusions,  # Empty for derived predicates
                children=children,
                is_and=predicate.is_and,
            )
        else:
            raise ValueError(f"Unknown predicate type: {type(predicate)}")

    def update_counts(self, state: WindowState, token: int, value: float) -> None:
        """
        Update counts for a token and value.

        For plain predicates, checks if token matches and updates count if value constraints are met.
        For derived predicates, recursively updates children.
        """
        if not self.children:
            # Plain predicate case
            # Check if this token is in our vocabulary
            if not any(token == t.item() for t in self.tokens):
                return

            # Initialize predicate count if needed
            if self.name not in state.predicate_counts:
                state.predicate_counts[self.name] = 0

            # Check value thresholds
            min_val, max_val = self.value_limits
            min_incl, max_incl = self.value_inclusions

            should_count = True

            # Check minimum threshold
            if min_val is not None:
                if min_incl:
                    should_count = value >= min_val
                else:
                    should_count = value > min_val

            # Check maximum threshold
            if should_count and max_val is not None:
                if max_incl:
                    should_count = value <= max_val
                else:
                    should_count = value < max_val

            if should_count:
                state.predicate_counts[self.name] += 1
        else:
            # Derived predicate case - update all children
            for child in self.children:
                child.update_counts(state, token, value)

    def get_count(self, state: WindowState) -> int:
        """
        Get total count for this predicate.

        For plain predicates, returns the stored count.
        For derived predicates, combines child counts according to AND/OR logic.
        """
        if not self.children:
            # Plain predicate - return stored count
            return state.predicate_counts.get(self.name, 0)

        # Derived predicate - combine child counts
        child_counts = [child.get_count(state) for child in self.children]

        if self.is_and:
            # AND - use minimum count
            return min(child_counts) if child_counts else 0
        else:
            # OR - use sum of counts
            return sum(child_counts)

    def check_constraints(self, state: WindowState, min_count, max_count) -> bool:
        """
        Check if count constraints are satisfied.
        """
        count = self.get_count(state)
        if min_count is not None and count < min_count:
            return False

        if max_count is not None and count > max_count:
            return False

        return True

    def check_impossible(self, state: WindowState, max_count) -> bool:
        """
        Check if constraints are impossible to satisfy.
        """
        count = self.get_count(state)

        if max_count is not None and count > max_count:
            return True

        return False


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
    batch_states: list[WindowState] = field(default_factory=list)
    ignore: bool = False
    label_value: bool | None = None

    def get_labels(self):
        labels = []
        labels.append(self.label_value)
        for node in self.children:
            labels.extend(node.get_labels())
        return labels

    def ignore_windows(self, window_names: list[str]):
        if self.name in window_names:
            self.ignore = True
            for each in self.batch_states:
                each.status = WindowStatus.SATISFIED
        for child in self.children:
            child.ignore_windows(window_names)

    def initialize_batch(self, batch_size: int):
        """Initialize batch states."""
        self.batch_states = [WindowState() for _ in range(batch_size)]
        # Also initialize children
        for child in self.children:
            child.initialize_batch(batch_size)

    def _check_label(self, state: WindowState) -> bool:
        if self.label is not None:
            if self._get_count(self.label, state) > 0:
                self.label_value = True

    def _check_start_condition(self, time_delta: float, event_token: int, batch_idx: int) -> bool:
        """Check if window should start at current time/event."""
        # Parse reference point
        ref_time = 0.0  # Default to trigger time

        # Check based on bound type
        if self.start_bound.bound_type == BoundType.TEMPORAL:
            offset_days = self.start_bound.offset.total_seconds() / (24 * 3600)  # Convert to days
            target_time = ref_time + offset_days

            if self.start_bound.inclusive:
                return time_delta >= target_time
            return time_delta > target_time

        else:  # EventBound
            if self.start_bound.direction == "next":
                # Check if this is the target event
                is_target = str(event_token) == self.start_bound.predicate
                at_or_after_ref = time_delta >= ref_time
                return is_target and at_or_after_ref
            else:  # previous
                # Not implemented - would need history
                raise NotImplementedError("Previous event bounds not yet supported")

    def _check_end_condition(self, time_delta: float, event_token: int) -> bool:
        """Check if window should end at current time/event."""
        # Parse reference point - similar to start condition
        ref_time = 0.0
        if self.end_bound.reference != "trigger":
            window_name, point = self.end_bound.reference.split(".")
            # Would need to look up reference window state
            # For simplicity, assume trigger reference for now

        # Check based on bound type
        if self.end_bound.bound_type == BoundType.TEMPORAL:
            offset_days = self.end_bound.offset.total_seconds() / (24 * 3600)
            target_time = ref_time + offset_days

            if self.end_bound.inclusive:
                return time_delta >= target_time
            return time_delta > target_time

        else:  # EventBound
            if self.end_bound.direction == "next":
                return str(event_token) == self.end_bound.predicate
            else:  # previous
                raise NotImplementedError("Previous event bounds not yet supported")

    def _update_counts(self, state: WindowState, event_token: int, numeric_value: float):
        """Update predicate counts for the window."""

        for _, tensorized_predicate in self.tensorized_predicates.items():
            tensorized_predicate.update_counts(state, event_token, numeric_value)

    def _get_count(self, predicate_id: str, state: WindowState) -> int:
        predicate_tensor = self.tensorized_predicates[predicate_id]
        return predicate_tensor.get_count(state)

    def _check_constraints_satisfied(self, state: WindowState) -> bool:
        """Check if all predicate constraints are satisfied."""
        for pred, (min_count, max_count) in self.predicate_constraints.items():
            count = self._get_count(pred, state)

            # Must meet minimum count
            if min_count is not None and count < min_count:
                return False
            # Must not exceed maximum count
            if max_count is not None and count > max_count:
                return False

        # For trigger window, constraints are satisfied as soon as met
        if self.name == "trigger":
            return True

        # For other windows, need to wait for window end
        if isinstance(self.end_bound, TemporalBound):
            return state.end_time is not None
        else:
            predicate = self.end_bound.predicate
            if isinstance(predicate, str):
                raise NotImplementedError("Non-Predicate window End Event Bound predicates not yet supported")
                # predicate = [int(self.end_bound.predicate)]
            return self.end_bound.predicate.check_constraints(state, 1, None)

    def _check_constraints_impossible(self, state: WindowState) -> bool:
        """Check if constraints are impossible to satisfy."""
        for pred, (min_count, max_count) in self.predicate_constraints.items():
            count = self._get_count(pred, state)

            # If we've exceeded max count
            if max_count is not None and count > max_count:
                return True

            # If window ended and we haven't hit min count
            if state.end_time and min_count is not None and count < min_count:
                return True

        return False

    def update(
        self,
        batch_idx: int,
        time_delta: float,
        event_token: int,
        numeric_values: float,
        parent_state: WindowState | None = None,
    ) -> WindowStatus:
        """Update state for specific batch element."""
        if self.ignore:
            return WindowStatus.SATISFIED
        state = self.batch_states[batch_idx]

        # If we were waiting for next time point to confirm window end
        if state.waiting_for_next_time is not None:
            if time_delta > state.waiting_for_next_time:
                state.status = WindowStatus.SATISFIED
                state.waiting_for_next_time = None
                self._check_label(state)
            else:
                self._update_counts(state, event_token, numeric_values)
            return state.status

        # If already satisfied or impossible, don't update
        if state.status in (WindowStatus.SATISFIED, WindowStatus.IMPOSSIBLE):
            return state.status

        # Check window start if not started
        if not state.start_time and self._check_start_condition(time_delta, event_token, batch_idx):
            state.start_time = time_delta
            state.in_window = True

        # Update counts if in window
        if state.in_window:
            self._update_counts(state, event_token, numeric_values)
            state.status = WindowStatus.ACTIVE

            self._check_label(state)

            # Check if constraints now impossible
            if self._check_constraints_impossible(state):
                state.status = WindowStatus.IMPOSSIBLE
                state.in_window = False
                return state.status

            # Check if constraints satisfied
            if self._check_constraints_satisfied(state) or self.label_value:
                if isinstance(self.end_bound, EventBound) and self.end_bound.inclusive:
                    # Wait for next time point to confirm end
                    state.waiting_for_next_time = time_delta
                    return state.status
                state.status = WindowStatus.SATISFIED
                state.in_window = False  # Close window once satisfied
                return state.status

        # Check window end
        if state.in_window and self._check_end_condition(time_delta, event_token):
            state.end_time = time_delta
            state.in_window = False

            # Final constraint check at window end
            if self._check_constraints_satisfied(state):
                state.status = WindowStatus.SATISFIED
            else:
                state.status = WindowStatus.IMPOSSIBLE

        return state.status


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

        def process_node(
            node: WindowNode, batch_idx: int, parent_state: WindowState | None = None
        ) -> WindowStatus:
            # Update this node's state
            status = node.update(
                batch_idx,
                time_deltas[batch_idx].item(),
                tokens[batch_idx].item(),
                numeric_values,
                parent_state,
            )

            # If this node is satisfied, process children
            if status == WindowStatus.SATISFIED:
                for child in node.children:
                    child_status = process_node(child, batch_idx, node.batch_states[batch_idx])
                    # Node only satisfied if all children satisfied
                    if child_status != WindowStatus.SATISFIED:
                        status = child_status
                        break

            return status

        # Process each sequence in batch
        results = []
        for i in range(self.batch_size):
            status = process_node(self.root, i)
            results.append(status)

        return torch.tensor([s.value for s in results])


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
