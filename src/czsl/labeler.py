from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Literal, Optional

import polars as pl
import torch
from loguru import logger

from czsl.config import (
    END_OF_RECORD_KEY,
    START_OF_RECORD_KEY,
    DerivedPredicateConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    TemporalWindowBounds,
    ToEventWindowBounds,
)


def get_predicate_tensor(metadata_df: pl.DataFrame, config: TaskExtractorConfig, predicate_name: str):
    predicate = config.predicates[predicate_name]
    if isinstance(predicate, PlainPredicateConfig):
        predicate_tensor = metadata_df.filter(predicate.MEDS_eval_expr())["code/vocab_index"].to_torch()
        predicate_value_limits = {
            str(each.item()): (predicate.value_min, predicate.value_max) for each in predicate_tensor
        }
        predicate_value_limit_inclusion = {
            str(each.item()): (predicate.value_min_inclusive, predicate.value_max_inclusive)
            for each in predicate_tensor
        }
    elif isinstance(predicate, DerivedPredicateConfig):
        if predicate.is_and:
            raise NotImplementedError("AND predicates not yet supported")
        predicates = predicate.input_predicates
        predicate_value_limits = {}
        predicate_value_limit_inclusion = {}
        p_tensors_list = []
        for p in predicates:
            p_tensors = metadata_df.filter(config.plain_predicates[p].MEDS_eval_expr())["code/vocab_index"]
            p_tensors_list.append(p_tensors.to_torch())
            predicate_value_limits.update(
                {
                    str(each): (
                        config.plain_predicates[p].value_min
                        if predicate_value_limits.get(str(each), (None, None))[0] is None
                        else predicate_value_limits[str(each)][0],
                        config.plain_predicates[p].value_max
                        if predicate_value_limits.get(str(each), (None, None))[1] is None
                        else predicate_value_limits[str(each)][1],
                    )
                    for each in p_tensors
                }
            )

            predicate_value_limit_inclusion.update(
                {
                    str(each): (
                        config.plain_predicates[p].value_min_inclusive
                        if predicate_value_limit_inclusion.get(str(each), (None, None))[0] is None
                        else predicate_value_limit_inclusion[str(each)][0],
                        config.plain_predicates[p].value_max_inclusive
                        if predicate_value_limit_inclusion.get(str(each), (None, None))[1] is None
                        else predicate_value_limit_inclusion[str(each)][1],
                    )
                    for each in p_tensors
                }
            )
        predicate_tensor = torch.concat(p_tensors_list)
    else:
        raise ValueError(f"Unknown predicate type {type(predicate)}")
    if predicate_tensor.shape[0] == 0:
        logger.warning(f"Predicate {predicate_name} returned no rows. Skipping it.")
    return predicate_tensor, predicate_value_limits, predicate_value_limit_inclusion


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
class EventBound(WindowBound):
    """Bound defined by occurrence of events."""

    predicate: str
    direction: Literal["next", "previous"]
    bound_type: BoundType = field(default=BoundType.EVENT, init=False)


@dataclass
class WindowState:
    """State of a window for one sequence in batch."""

    start_time: float | None = None
    end_time: float | None = None
    in_window: bool = False
    predicate_counts: dict[str, int] = field(default_factory=dict)
    status: WindowStatus = WindowStatus.UNDETERMINED

    def reset(self):
        """Reset state for new sequence."""
        self.start_time = None
        self.end_time = None
        self.in_window = False
        self.predicate_counts.clear()
        self.status = WindowStatus.UNDETERMINED


T = None | int


@dataclass
class WindowNode:
    """Node in autoregressive window tree."""

    name: str
    start_bound: TemporalBound | EventBound
    end_bound: TemporalBound | EventBound
    predicate_constraints: dict[str, tuple[int | None, int | None]]
    label: str | None
    index_timestamp: str | None
    tensorized_predicates: dict[str, torch.Tensor]
    predicate_value_limits: dict[str, tuple[(int | None), (int | None)]]
    predicate_value_limit_inclusion: dict[str, tuple[(bool | None), (bool | None)]]
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
        token_str = str(event_token)

        if token_str not in state.predicate_counts:
            state.predicate_counts[token_str] = 0

        logger.warning(f"numeric_value limit is set for all vocab_indices {token_str} in window {self.name}")

        if token_str in self.predicate_value_limits:
            min_value, max_value = self.predicate_value_limits[token_str]
            if min_value is not None or max_value is not None:
                min_inclusive, max_inclusive = self.predicate_value_limit_inclusion[token_str]
                if min_value is not None and (
                    numeric_value > min_value or (min_inclusive and numeric_value == min_value)
                ):
                    # high numeric value check
                    state.predicate_counts[token_str] += 1
                if max_value is not None and (
                    numeric_value < max_value or (max_inclusive and numeric_value == max_value)
                ):
                    # low numeric value check
                    state.predicate_counts[token_str] += 1
            else:
                state.predicate_counts[token_str] += 1
        else:
            state.predicate_counts[token_str] += 1

    def _get_count(self, predicate_id: str, state: WindowState) -> int:
        predicate_tensor = self.tensorized_predicates[predicate_id]
        count = 0
        for each in predicate_tensor:
            count += state.predicate_counts.get(str(each.item()), 0)
        return count

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
            if isinstance(self.end_bound.predicate, str):
                predicate = [int(self.end_bound.predicate)]
            else:
                predicate = self.end_bound.predicate
            return any([int(event_token) in predicate for event_token in state.predicate_counts.keys()])

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

        logger.info(f"Node {self.name} updating batch {batch_idx}:")
        logger.info(f"  token: {event_token}, time: {time_delta}")
        logger.info(
            f"  current state: start={state.start_time}, end={state.end_time}, in_window={state.in_window}"
        )

        # If already satisfied or impossible, don't update
        if state.status in (WindowStatus.SATISFIED, WindowStatus.IMPOSSIBLE):
            logger.info(f"  skipping - already {state.status}")
            return state.status

        # Check window start if not started
        if not state.start_time and self._check_start_condition(time_delta, event_token, batch_idx):
            state.start_time = time_delta
            state.in_window = True
            logger.info(f"  window started at {time_delta}")

        # Update counts if in window
        if state.in_window:
            self._update_counts(state, event_token, numeric_values)
            state.status = WindowStatus.ACTIVE
            logger.info(f"  updated counts: {state.predicate_counts}")

            self._check_label(state)

            # Check if constraints now impossible
            if self._check_constraints_impossible(state):
                state.status = WindowStatus.IMPOSSIBLE
                state.in_window = False
                logger.info("  constraints impossible")
                return state.status

            # Check if constraints satisfied
            if self._check_constraints_satisfied(state) or self.label_value:
                state.status = WindowStatus.SATISFIED
                state.in_window = False  # Close window once satisfied
                logger.info("  constraints satisfied")
                return state.status

        # Check window end
        if state.in_window and self._check_end_condition(time_delta, event_token):
            state.end_time = time_delta
            state.in_window = False
            logger.info(f"  window ended at {time_delta}")

            # Final constraint check at window end
            if self._check_constraints_satisfied(state):
                state.status = WindowStatus.SATISFIED
                logger.info("  constraints satisfied")
            else:
                state.status = WindowStatus.IMPOSSIBLE
                logger.info("  constraints impossible - window ended")

        logger.info(f"  final status: {state.status}")
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
    tensorized_predicates, predicate_value_limits, predicate_value_limit_inclusion = {}, {}, {}
    for p in config.predicates:
        pred_tensor, pred_value_limit, pred_value_limit_inclusion = get_predicate_tensor(
            metadata_df, config, p
        )
        tensorized_predicates[p] = pred_tensor
        predicate_value_limits.update(pred_value_limit)
        predicate_value_limit_inclusion.update(pred_value_limit_inclusion)

    # 1. Create trigger/root node
    root = WindowNode(
        name="trigger",
        start_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        end_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        predicate_constraints={config.trigger.predicate: (1, 1)},
        label=None,
        index_timestamp=None,
        tensorized_predicates=tensorized_predicates,
        predicate_value_limits=predicate_value_limits,
        predicate_value_limit_inclusion=predicate_value_limit_inclusion,
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
                predicate, _, _ = get_predicate_tensor(metadata_df, config, predicate)

            return (
                EventBound(
                    reference=window_name,
                    inclusive=expr.left_inclusive,
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
            predicate_value_limits=predicate_value_limits,
            predicate_value_limit_inclusion=predicate_value_limit_inclusion,
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
