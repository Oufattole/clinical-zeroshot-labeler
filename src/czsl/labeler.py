from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Literal, Optional

import torch


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


@dataclass
class WindowNode:
    """Node in autoregressive window tree."""

    name: str
    start_bound: TemporalBound | EventBound
    end_bound: TemporalBound | EventBound
    predicate_constraints: dict[str, tuple[int | None, int | None]]
    parent: Optional["WindowNode"] = None
    children: list["WindowNode"] = field(default_factory=list)
    batch_states: list[WindowState] = field(default_factory=list)

    def initialize_batch(self, batch_size: int):
        """Initialize batch states."""
        self.batch_states = [WindowState() for _ in range(batch_size)]
        # Also initialize children
        for child in self.children:
            child.initialize_batch(batch_size)

    def _check_start_condition(
        self, time_delta: float, event_token: int, parent_state: WindowState | None = None
    ) -> bool:
        """Check if window should start at current time/event."""
        # Parse reference point
        ref_time = 0.0  # Default to trigger time
        if self.start_bound.reference != "trigger":
            if not parent_state:
                return False
            # Get time from parent window
            window_name, point = self.start_bound.reference.split(".")
            if point == "start":
                ref_time = parent_state.start_time
            else:  # end
                ref_time = parent_state.end_time

            if ref_time is None:
                return False

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

    def _update_counts(self, state: WindowState, event_token: int):
        """Update predicate counts for the window."""
        token_str = str(event_token)
        if token_str not in state.predicate_counts:
            state.predicate_counts[token_str] = 0
        state.predicate_counts[token_str] += 1

    def _check_constraints_satisfied(self, state: WindowState) -> bool:
        """Check if all predicate constraints are satisfied."""
        for pred, (min_count, max_count) in self.predicate_constraints.items():
            count = state.predicate_counts.get(pred, 0)

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
        return state.end_time is not None

    def _check_constraints_impossible(self, state: WindowState) -> bool:
        """Check if constraints are impossible to satisfy."""
        for pred, (min_count, max_count) in self.predicate_constraints.items():
            count = state.predicate_counts.get(pred, 0)

            # If we've exceeded max count
            if max_count is not None and count > max_count:
                return True

            # If window ended and we haven't hit min count
            if state.end_time and min_count is not None and count < min_count:
                return True

        return False

    def update(
        self, batch_idx: int, time_delta: float, event_token: int, parent_state: WindowState | None = None
    ) -> WindowStatus:
        """Update state for specific batch element."""
        state = self.batch_states[batch_idx]

        print(f"Node {self.name} updating batch {batch_idx}:")
        print(f"  token: {event_token}, time: {time_delta}")
        print(f"  current state: start={state.start_time}, end={state.end_time}, in_window={state.in_window}")

        # If already satisfied or impossible, don't update
        if state.status in (WindowStatus.SATISFIED, WindowStatus.IMPOSSIBLE):
            print(f"  skipping - already {state.status}")
            return state.status

        # Check window start if not started
        if not state.start_time and self._check_start_condition(time_delta, event_token, parent_state):
            state.start_time = time_delta
            state.in_window = True
            print(f"  window started at {time_delta}")

        # Update counts if in window
        if state.in_window:
            self._update_counts(state, event_token)
            state.status = WindowStatus.ACTIVE
            print(f"  updated counts: {state.predicate_counts}")

            # Check if constraints satisfied
            if self._check_constraints_satisfied(state):
                state.status = WindowStatus.SATISFIED
                state.in_window = False  # Close window once satisfied
                print("  constraints satisfied")
                return state.status

            # Check if constraints now impossible
            if self._check_constraints_impossible(state):
                state.status = WindowStatus.IMPOSSIBLE
                state.in_window = False
                print("  constraints impossible")
                return state.status

        # Check window end
        if state.in_window and self._check_end_condition(time_delta, event_token):
            state.end_time = time_delta
            state.in_window = False
            print(f"  window ended at {time_delta}")

            # Final constraint check at window end
            if self._check_constraints_satisfied(state):
                state.status = WindowStatus.SATISFIED
                print("  constraints satisfied")
            else:
                state.status = WindowStatus.IMPOSSIBLE
                print("  constraints impossible - window ended")

        print(f"  final status: {state.status}")
        return state.status


class AutoregressiveWindowTree:
    """Manages window tree for autoregressive constraint tracking."""

    def __init__(self, root: WindowNode, batch_size: int):
        self.root = root
        self.batch_size = batch_size
        # Initialize states for all nodes
        self.root.initialize_batch(batch_size)

    def update(
        self, tokens: torch.Tensor, time_deltas: torch.Tensor  # [batch_size]  # [batch_size]
    ) -> torch.Tensor:  # [batch_size] of ConstraintStatus
        """Process new tokens through tree."""

        def process_node(
            node: WindowNode, batch_idx: int, parent_state: WindowState | None = None
        ) -> WindowStatus:
            # Update this node's state
            status = node.update(
                batch_idx, time_deltas[batch_idx].item(), tokens[batch_idx].item(), parent_state
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
