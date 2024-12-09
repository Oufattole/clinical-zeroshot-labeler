from dataclasses import InitVar, dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import polars as pl
import torch
from bigtree import yield_tree
from loguru import logger

from czsl.config import (
    DerivedPredicateConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    TemporalWindowBounds,
    ToEventWindowBounds,
    WindowConfig,
)
from czsl.types import END_OF_RECORD_KEY, START_OF_RECORD_KEY

TIME_SCALE = "Y"


@dataclass
class GenerationBudget:
    """Optional constraints for generation length and time."""

    max_seq_len: int | None = None
    max_time: float | None = None

    def __post_init__(self):
        if self.max_seq_len is not None and self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.max_time is not None and self.max_time <= 0:
            raise ValueError(f"max_time must be positive, got {self.max_time}")


class ConstraintStatus(Enum):
    """Status of a constraint evaluation."""

    UNDETERMINED = 0
    SATISFIED = 1
    IMPOSSIBLE = 2


@dataclass
class TemporalBoundTracker:
    """Tracks time bounds for a batch of sequences."""

    bound: InitVar[TemporalWindowBounds]
    is_start: bool
    predicate_configs: dict[str, PlainPredicateConfig]
    left: torch.Tensor = field(init=False)
    left_inclusive: bool = field(init=False)
    right: torch.Tensor = field(init=False)
    right_inclusive: bool = field(init=False)

    def __post_init__(self, bound):
        self.left = np.timedelta64(bound.offset) / np.timedelta64(1, TIME_SCALE).astype("timedelta64[us]")
        self.left_inclusive = bound.left_inclusive
        self.right = np.timedelta64(bound.window_size) / np.timedelta64(1, TIME_SCALE).astype(
            "timedelta64[us]"
        )
        self.right_inclusive = bound.right_inclusive

    def is_in(self, time_ref: torch.Tensor, event_tokens: torch.Tensor) -> torch.Tensor:
        """Update time constraint status.

        Args:
            time_ref: Current time relative to prediction time

        Returns:
            Updated constraint status
        """
        in_window = (time_ref >= self.left) & (time_ref <= self.right)
        if self.left_inclusive:
            in_window = in_window | (time_ref == self.left)
        if self.right_inclusive:
            in_window = in_window | (time_ref == self.right)
        return in_window


@dataclass
class PlainPredicateEvaluator:
    """Evaluates plain predicates that directly map to tokens.

    Args:
        token_ids: Set of token IDs that represent this predicate
    """

    token_ids: set[int]

    def evaluate(self, event_tokens: torch.Tensor) -> torch.Tensor:
        """Check if tokens match any of the predicate's tokens.

        Args:
            event_tokens: Tensor of shape (batch_size,) with token IDs

        Returns:
            Boolean tensor of shape (batch_size,) indicating matches
        """
        return torch.isin(event_tokens, torch.tensor(list(self.token_ids), device=event_tokens.device))


@dataclass
class AndPredicateEvaluator:
    """Evaluates AND of multiple predicates.

    Args:
        evaluators: List of predicate evaluators to AND together
    """

    evaluators: list[PlainPredicateEvaluator]

    def evaluate(self, event_tokens: torch.Tensor) -> torch.Tensor:
        """Check if tokens satisfy all sub-predicates.

        Args:
            event_tokens: Tensor of shape (batch_size,) with token IDs

        Returns:
            Boolean tensor of shape (batch_size,) indicating matches
        """
        # Start with all True and AND with each evaluator
        result = torch.ones(event_tokens.shape[0], dtype=torch.bool, device=event_tokens.device)
        for evaluator in self.evaluators:
            result &= evaluator.evaluate(event_tokens)
        return result


@dataclass
class OrPredicateEvaluator:
    """Evaluates OR of multiple predicates.

    Args:
        evaluators: List of predicate evaluators to OR together
    """

    evaluators: list[PlainPredicateEvaluator]

    def evaluate(self, event_tokens: torch.Tensor) -> torch.Tensor:
        """Check if tokens satisfy any sub-predicates.

        Args:
            event_tokens: Tensor of shape (batch_size,) with token IDs

        Returns:
            Boolean tensor of shape (batch_size,) indicating matches
        """
        # Start with all False and OR with each evaluator
        result = torch.zeros(event_tokens.shape[0], dtype=torch.bool, device=event_tokens.device)
        for evaluator in self.evaluators:
            result |= evaluator.evaluate(event_tokens)
        return result


@dataclass
class PredicateTracker:
    """Tracks predicate counts and constraints for a window.

    Args:
        evaluator: Predicate evaluator (Plain, And, or Or)
        min_count: Minimum required occurrences (None for no minimum)
        max_count: Maximum allowed occurrences (None for no maximum)
        batch_size: Size of batch being tracked
        device: Device for torch tensors
    """

    evaluator: PlainPredicateEvaluator | AndPredicateEvaluator | OrPredicateEvaluator
    min_count: int | None
    max_count: int | None
    batch_size: int
    device: str = "cpu"

    # Internal state
    counts: torch.Tensor = field(init=False)
    status: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.counts = torch.zeros(self.batch_size, device=self.device)
        self.status = torch.full((self.batch_size,), ConstraintStatus.UNDETERMINED.value, device=self.device)

    def update(self, tokens: torch.Tensor) -> torch.Tensor:
        """Update counts for new tokens and return constraint status.

        Args:
            tokens: Tensor of shape (batch_size,) with token IDs

        Returns:
            Tensor of shape (batch_size,) with ConstraintStatus values
        """
        # Update counts where predicate matches
        matches = self.evaluator.evaluate(tokens)
        self.counts[matches] += 1

        # Check if constraints are impossible or satisfied
        if self.max_count is not None:
            impossible = self.counts > self.max_count
            self.status = torch.where(
                impossible, torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=self.device), self.status
            )

        if self.min_count is not None:
            satisfied = self.counts >= self.min_count
            not_impossible = self.status != ConstraintStatus.IMPOSSIBLE.value
            self.status = torch.where(
                satisfied & not_impossible,
                torch.tensor(ConstraintStatus.SATISFIED.value, device=self.device),
                self.status,
            )

        return self.status


def create_predicate_evaluator(
    predicate_config: PlainPredicateConfig | DerivedPredicateConfig,
    token_to_code_map: dict[int, str],
    predicate_configs: dict[str, PlainPredicateConfig],
) -> PlainPredicateEvaluator | AndPredicateEvaluator | OrPredicateEvaluator:
    """Create appropriate predicate evaluator from config.

    Args:
        predicate_config: Plain or derived predicate configuration
        token_to_code_map: Mapping from token IDs to predicate codes

    Returns:
        Configured predicate evaluator
    """
    if isinstance(predicate_config, PlainPredicateConfig):
        # Find all token IDs that map to this predicate's code
        token_ids = {token for token, code in token_to_code_map.items() if code == predicate_config.code}
        return PlainPredicateEvaluator(token_ids)

    elif isinstance(predicate_config, DerivedPredicateConfig):
        # Parse expression and get sub-predicate names
        expr = predicate_config.expr
        pred_names = predicate_config.input_predicates
        evaluators = [
            create_predicate_evaluator(predicate_configs[name], token_to_code_map, predicate_configs)
            for name in pred_names
        ]
        if expr.startswith("and("):
            return AndPredicateEvaluator(evaluators)
        elif expr.startswith("or("):
            return OrPredicateEvaluator(evaluators)

        else:
            raise ValueError(f"Invalid predicate expression: {expr}")

    else:
        raise ValueError(f"Invalid predicate config type: {type(predicate_config)}")


@dataclass
class ToEventBoundTracker:
    """Tracks time bounds for a batch of sequences based on event occurrences.

    This tracker determines if a time point is within bounds by looking for specific events
    that occur relative to a reference point. For end bounds, looks forward for first matching
    event. For start bounds, looks backward for last matching event.

    For the special event START_OF_RECORD_KEY and END_OF_RECORD_KEY constraints are always satisfied
    as long as they are bounding the start and end of the window respectively.

    Args:
        bound: The event-based window bounds configuration
        is_start: Whether this tracks the start (True) or end (False) of a window
        batch_size: Size of the batch being processed
        token_to_code_map: Mapping from token IDs to predicate codes
        predicate_config: Configuration for the event predicate to match
        device: Device to place tensors on

    Examples:
        >>> import torch
        >>> from datetime import timedelta
        >>> # Create token mapping
        >>> token_map = {
        ...     1: "DISCHARGE",
        ...     2: "DEATH",
        ...     3: "LAB_TEST"
        ... }
        >>> # Create bounds configuration
        >>> bounds = ToEventWindowBounds(
        ...     left_inclusive=False,
        ...     end_event='death_or_discharge',
        ...     right_inclusive=True,
        ...     offset=timedelta(0)
        ... )
        >>> # Set up predicate config for discharge_or_death
        >>> predicate_config = DerivedPredicateConfig(expr="or(death, discharge)")
        >>> # Add the predicate configs it depends on
        >>> predicate_config.predicates = {
        ...    "death": PlainPredicateConfig(code="DEATH"),
        ...    "discharge": PlainPredicateConfig(code="DISCHARGE")
        ... }
        >>> # Create end tracker (looking forward for events)
        >>> tracker = ToEventBoundTracker(
        ...    bounds, False, 2, token_map, predicate_config, predicate_configs=predicate_config.predicates,
        ... )
        >>> # Initially status should be undetermined
        >>> tracker.status
        tensor([0, 0])
        >>> # Test with some time points and events
        >>> # Sequence 1: death (token 2) occurs at t=5 after reference t=0
        >>> # Sequence 2: no death/discharge occurs
        >>> times = torch.tensor([0.0, 0.0])  # Reference time = 0
        >>> events = torch.tensor([3, 3])  # LAB_TEST tokens
        >>> tracker.is_in(times, events)  # No events found yet
        tensor([0, 0])
        >>> times = torch.tensor([5.0, 5.0])  # 5 hours after reference
        >>> events = torch.tensor([2, 3])  # First sequence has death token
        >>> tracker.is_in(times, events)  # First sequence finds valid future event
        tensor([1, 0])
        >>> times = torch.tensor([10.0, 10.0])
        >>> events = torch.tensor([3, 3])  # More lab tests
        >>> tracker.is_in(times, events)  # Status remains the same
        tensor([1, 0])
        >>> # Create start tracker (looking backward for events)
        >>> tracker = ToEventBoundTracker(
        ...    bounds, True, 2, token_map, predicate_config, predicate_configs=predicate_config.predicates,
        ... )
        >>> times = torch.tensor([-5.0, -5.0])  # Before reference time
        >>> events = torch.tensor([2, 3])  # First sequence has death token
        >>> tracker.is_in(times, events)  # First sequence finds valid past event
        tensor([1, 0])
    """

    bound: InitVar[ToEventWindowBounds]
    is_start: bool  # True for start bound, False for end bound
    batch_size: int
    token_to_code_map: dict[int, str]
    predicate_config: PlainPredicateConfig | DerivedPredicateConfig | str
    predicate_configs: dict[str, PlainPredicateConfig]
    device: str = "cpu"

    # Internal state
    status: torch.Tensor = field(init=False)  # Current constraint status
    evaluator: PlainPredicateEvaluator | AndPredicateEvaluator | OrPredicateEvaluator | None = None
    left_inclusive: bool = field(init=False)
    right_inclusive: bool = field(init=False)

    def __post_init__(self, bound: ToEventWindowBounds):
        # Store bound inclusivity settings
        self.left_inclusive = bound.left_inclusive
        self.right_inclusive = bound.right_inclusive

        # Initialize status as undetermined for all sequences
        self.status = torch.full((self.batch_size,), ConstraintStatus.UNDETERMINED.value, device=self.device)

        # Create the predicate evaluator
        if self.predicate_config == START_OF_RECORD_KEY or self.predicate_config == END_OF_RECORD_KEY:
            self.evaluator = None
        else:
            self.evaluator = create_predicate_evaluator(
                self.predicate_config,
                self.token_to_code_map,
                self.predicate_configs,
            )

    def is_in(self, time_ref: torch.Tensor, event_tokens: torch.Tensor) -> torch.Tensor:
        """Update time constraint status based on event occurrences.

        Args:
            time_ref: Time relative to reference point, shape (batch_size,)
                For end bounds: time since reference (valid if > 0)
                For start bounds: time before reference (valid if < 0)
            event_tokens: Event tokens for the current position, shape (batch_size,)
                These should be the raw token IDs

        Returns:
            Updated constraint status for each sequence in batch
        """
        if self.predicate_config == START_OF_RECORD_KEY:
            assert self.is_start, "START_OF_RECORD_KEY can always be at the start of the window"
            return torch.full_like(event_tokens, ConstraintStatus.SATISFIED.value)
        if self.predicate_config == END_OF_RECORD_KEY:
            assert not self.is_start, "END_OF_RECORD_KEY can always be at the end of the window"
            return torch.full_like(event_tokens, ConstraintStatus.SATISFIED.value)
        # Check if current tokens match our target predicate
        is_target_event = self.evaluator.evaluate(event_tokens)

        # Validate times relative to reference point
        if self.is_start:
            # For start bounds, time must be before reference (negative)
            valid_times = time_ref < 0
            if self.left_inclusive:
                valid_times |= time_ref == 0
        else:
            # For end bounds, time must be after reference (positive)
            valid_times = time_ref > 0
            if self.right_inclusive:
                valid_times |= time_ref == 0

        # Update status where we find valid event occurrences
        event_found = is_target_event & valid_times
        self.status = torch.where(event_found, ConstraintStatus.SATISFIED.value, self.status)

        return self.status


def create_time_bound_traker(
    is_start: bool,
    endpoint_expr: TemporalWindowBounds | ToEventWindowBounds,
    batch_size: int | None = None,
    token_to_code_map: dict[int, str] | None = None,
    predicate_config: PlainPredicateConfig | DerivedPredicateConfig | str | None = None,
    predicate_configs: dict[str, PlainPredicateConfig] = None,
    device: str = "cpu",
) -> TemporalBoundTracker | ToEventBoundTracker:
    """Create appropriate time bound tracker based on endpoint expression type.

    Args:
        is_start: Whether this tracks start (True) or end (False) of window
        endpoint_expr: Time window bounds configuration
        batch_size: Required for ToEventBoundTracker
        token_to_code_map: Required for ToEventBoundTracker
        predicate_config: Required for ToEventBoundTracker if using event bounds
        device: Device for torch tensors

    Returns:
        Appropriate tracker for the endpoint type

    Raises:
        ValueError: If ToEventBoundTracker parameters are missing
    """
    if isinstance(endpoint_expr, TemporalWindowBounds):
        return TemporalBoundTracker(endpoint_expr, is_start, predicate_configs)
    else:
        if batch_size is None:
            raise ValueError("ToEventBoundTracker requires batch_size")
        if token_to_code_map is None:
            raise ValueError("ToEventBoundTracker requires token_to_code_map")
        if predicate_config is None:
            raise ValueError("ToEventBoundTracker requires predicate_config")

        return ToEventBoundTracker(
            bound=endpoint_expr,
            is_start=is_start,
            batch_size=batch_size,
            token_to_code_map=token_to_code_map,
            predicate_config=predicate_config,
            predicate_configs=predicate_configs,
            device=device,
        )


@dataclass
class WindowConstraints:
    """Tracks constraints for a window.

    Handles both time bounds and predicate constraints within windows.
    Time bounds can be either temporal (fixed time windows) or event-based.
    Predicate constraints track required event occurrences within valid windows.

    >>> import torch
    >>> from datetime import timedelta
    >>> from dataclasses import dataclass

    # First set up our token mapping and predicate configs similar to a real medical task
    >>> token_map = {
    ...     0: "ICU_ADMIT",      # ICU admission
    ...     1: "DEATH",          # Death event
    ...     2: "DISCHARGE",      # Hospital discharge
    ...     3: "LAB_TEST",       # Lab measurements
    ...     4: "PROCEDURE"       # Medical procedures
    ... }

    # Set up predicate configurations
    >>> # Basic events
    >>> icu_pred = PlainPredicateConfig(code="ICU_ADMIT")
    >>> death_pred = PlainPredicateConfig(code="DEATH")
    >>> discharge_pred = PlainPredicateConfig(code="DISCHARGE")
    >>> lab_pred = PlainPredicateConfig(code="LAB_TEST")
    >>> procedure_pred = PlainPredicateConfig(code="PROCEDURE")
    >>> # Composite events
    >>> death_discharge_pred = DerivedPredicateConfig(expr="or(death, discharge)")
    >>> all_interventions_pred = DerivedPredicateConfig(expr="or(lab_test, procedure)")
    >>> # Create predicate config dictionary
    >>> predicate_configs = {
    ...     "icu_admit": icu_pred,
    ...     "death": death_pred,
    ...     "discharge": discharge_pred,
    ...     "lab_test": lab_pred,
    ...     "procedure": procedure_pred,
    ...     "death_or_discharge": death_discharge_pred,
    ...     "interventions": all_interventions_pred
    ... }

    # Test Case 1: Temporal window (first 24h after ICU)
    >>> # Window config for first 24h requiring at least 2 interventions
    >>> temporal_window = WindowConfig(
    ...     start=None,
    ...     end="trigger + 24h",
    ...     start_inclusive=True,
    ...     end_inclusive=True,
    ...     has={"interventions": "(2, None)"}  # At least 2 interventions
    ... )
    >>> # Create constraints tracker
    >>> temp_constraints = WindowConstraints(
    ...     window_config=temporal_window,
    ...     token_to_code_map=token_map,
    ...     predicate_configs=predicate_configs,
    ...     batch_size=2,
    ...     device="cpu"
    ... )
    >>> # Check initial status
    >>> temp_constraints.status
    tensor([0, 0])
    >>> # Update with events (both sequences get a lab test)
    >>> temp_constraints.update(torch.tensor([3, 3]), torch.tensor([5.0, 5.0]))
    tensor([0, 0])
    >>> # Still need more interventions
    >>> # More events: seq 1 gets procedure, seq 2 gets another lab
    >>> temp_constraints.update(torch.tensor([4, 3]), torch.tensor([10.0, 10.0]))
    tensor([1, 1])
    >>> # Both satisfied - have 2 interventions each

    # Test Case 2: Event-bounded window (ICU to death/discharge)
    >>> # Window config for ICU stay ending in death/discharge with constraints
    >>> event_window = WindowConfig(
    ...     start="trigger",
    ...     end="start -> death_or_discharge",
    ...     start_inclusive=True,
    ...     end_inclusive=True,
    ...     has={
    ...         "lab_test": "(1, 5)",      # 1-5 lab tests
    ...         "procedure": "(None, 2)"    # At most 2 procedures
    ...     }
    ... )
    >>> # Create constraints tracker
    >>> event_constraints = WindowConstraints(
    ...     window_config=event_window,
    ...     token_to_code_map=token_map,
    ...     predicate_configs=predicate_configs,
    ...     batch_size=2,
    ...     device="cpu"
    ... )
    >>> # Check initial status
    >>> event_constraints.status
    tensor([0, 0])
    >>> # Sequence of events for two patients:
    >>> # Patient 1: 2 labs, 1 procedure, discharge (valid)
    >>> # Patient 2: 6 labs, 3 procedures, death (exceeds limits)
    >>> # First lab tests
    >>> event_constraints.update(torch.tensor([3, 3]), torch.tensor([2.0, 2.0]))
    tensor([0, 0])
    >>> # More labs for patient 2
    >>> event_constraints.update(torch.tensor([0, 3]), torch.tensor([4.0, 4.0]))
    tensor([0, 0])
    >>> # Procedures
    >>> event_constraints.update(torch.tensor([4, 4]), torch.tensor([6.0, 6.0]))
    tensor([0, 0])
    >>> # More events for patient 2
    >>> event_constraints.update(torch.tensor([3, 4]), torch.tensor([8.0, 8.0]))
    tensor([0, 2])  # Patient 2 now impossible (too many procedures)
    >>> # Final events
    >>> event_constraints.update(torch.tensor([2, 1]), torch.tensor([10.0, 10.0]))
    tensor([1, 2])  # Patient 1 satisfied, Patient 2 remains impossible

    # Test Case 3: Empty predicate constraints
    >>> # Window that just needs to find death/discharge
    >>> simple_window = WindowConfig(
    ...     start="trigger",
    ...     end="start -> death_or_discharge",
    ...     start_inclusive=True,
    ...     end_inclusive=True,
    ...     has={}  # No additional constraints
    ... )
    >>> # Create constraints
    >>> simple_constraints = WindowConstraints(
    ...     window_config=simple_window,
    ...     token_to_code_map=token_map,
    ...     predicate_configs=predicate_configs,
    ...     batch_size=2,
    ...     device="cpu"
    ... )
    >>> # Just need to find death or discharge
    >>> simple_constraints.update(torch.tensor([3, 3]), torch.tensor([5.0, 5.0]))
    tensor([0, 0])
    >>> simple_constraints.update(torch.tensor([2, 1]), torch.tensor([10.0, 10.0]))
    tensor([1, 1])  # Both satisfied by finding valid end events
    """

    window_config: WindowConfig
    token_to_code_map: dict[int, str]
    predicate_configs: dict[str, PlainPredicateConfig | DerivedPredicateConfig]
    batch_size: InitVar[int]
    device: InitVar[str] = "cpu"
    status: torch.Tensor = field(init=False)

    def __post_init__(self, batch_size: int, device: str):
        # Create trackers for start/end bounds
        left_predicate_config = None
        right_predicate_config = None
        if isinstance(self.window_config.start_endpoint_expr, ToEventWindowBounds):
            left_predicate_name = self.window_config.start_endpoint_expr.end_event.replace("-", "")
            left_predicate_config = self.predicate_configs.get(left_predicate_name, left_predicate_name)
        if isinstance(self.window_config.end_endpoint_expr, ToEventWindowBounds):
            right_predicate_name = self.window_config.end_endpoint_expr.end_event.replace("-", "")
            right_predicate_config = self.predicate_configs.get(right_predicate_name, right_predicate_name)
        self.left_bound = create_time_bound_traker(
            True,
            self.window_config.start_endpoint_expr,
            batch_size,
            self.token_to_code_map,
            left_predicate_config,
            self.predicate_configs,
            device,
        )
        self.right_bound = create_time_bound_traker(
            False,
            self.window_config.end_endpoint_expr,
            batch_size,
            self.token_to_code_map,
            right_predicate_config,
            self.predicate_configs,
            device,
        )

        # Create trackers for predicate constraints within the window
        self.predicate_trackers = {}
        for pred_name, (min_count, max_count) in self.window_config.has.items():
            predicate_config = self.predicate_configs[pred_name]
            evaluator = create_predicate_evaluator(
                predicate_config, self.token_to_code_map, self.predicate_configs
            )
            tracker = PredicateTracker(
                evaluator=evaluator,
                min_count=min_count,
                max_count=max_count,
                batch_size=batch_size,
                device=device,
            )
            self.predicate_trackers[pred_name] = tracker

        self.status = torch.full((batch_size,), ConstraintStatus.UNDETERMINED.value, device=device)

    def update(self, tokens: torch.Tensor, time_ref: torch.Tensor) -> torch.Tensor:
        """Update all constraints and return overall status.

        Args:
            tokens: Current token IDs, shape (batch_size,)
            time_ref: Current time relative to reference point, shape (batch_size,)

        Returns:
            Status tensor of shape (batch_size,) with ConstraintStatus values
        """
        # Check time bounds first - need both bounds satisfied
        left_status = self.left_bound.is_in(time_ref, tokens)
        right_status = self.right_bound.is_in(time_ref, tokens)

        # No predicates to check - just need valid time bounds
        if not self.window_config.has:
            return torch.where(
                (left_status == ConstraintStatus.SATISFIED.value)
                & (right_status == ConstraintStatus.SATISFIED.value),
                ConstraintStatus.SATISFIED.value,
                ConstraintStatus.UNDETERMINED.value,
            )

        # Check if we're in valid time window
        time_ok = (left_status == ConstraintStatus.SATISFIED.value) & (
            right_status == ConstraintStatus.SATISFIED.value
        )

        if not time_ok.any():
            # Return current status if no sequences in valid time window
            return self.status

        # Update predicate trackers for sequences in valid time window
        pred_status = torch.stack(
            [tracker.update(tokens[time_ok]) for tracker in self.predicate_trackers.values()]
        )

        # Combine predicate statuses:
        # - Any impossible -> impossible
        # - All satisfied -> satisfied
        # - Otherwise -> undetermined
        impossible = (pred_status == ConstraintStatus.IMPOSSIBLE.value).any(dim=0)
        satisfied = (pred_status == ConstraintStatus.SATISFIED.value).all(dim=0)

        self.status[time_ok] = torch.where(
            impossible,
            torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=self.device),
            torch.where(
                satisfied,
                torch.tensor(ConstraintStatus.SATISFIED.value, device=self.device),
                torch.tensor(ConstraintStatus.UNDETERMINED.value, device=self.device),
            ),
        )

        return self.status


class BatchAutoregressiveLabeler:
    """Batch implementation of autoregressive task labeling."""

    def __init__(
        self,
        task_config: TaskExtractorConfig,
        token_to_code_map: dict[int, str],
        batch_size: int,
        device: str = "cpu",
    ):
        self.task_config = task_config
        self.token_to_code_map = token_to_code_map
        self.batch_size = batch_size
        self.device = device

        # Create window constraints
        self.constraints = {}

        for name, window in task_config.windows.items():
            constraints = WindowConstraints(
                window_config=window,
                batch_size=batch_size,
                token_to_code_map=token_to_code_map,
                device=device,
            )
            self.constraints[name] = constraints

    def update(
        self,
        next_tokens: torch.Tensor,
        next_times: torch.Tensor,
    ) -> torch.Tensor:
        """Update with new tokens and return status."""
        # Get status from each window
        results = torch.stack(
            [constraints.update(next_tokens, next_times) for constraints in self.constraints.values()]
        )

        # Combine window statuses:
        # - Any impossible -> impossible
        # - All satisfied -> satisfied
        # - Otherwise -> undetermined
        status = torch.full_like(next_tokens, ConstraintStatus.UNDETERMINED.value, device=self.device)

        if len(results) > 0:
            impossible = (results == ConstraintStatus.IMPOSSIBLE.value).any(dim=0)
            satisfied = (results == ConstraintStatus.SATISFIED.value).all(dim=0)

            status = torch.where(
                impossible,
                torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=self.device),
                torch.where(
                    satisfied,
                    torch.tensor(ConstraintStatus.SATISFIED.value, device=self.device),
                    torch.tensor(ConstraintStatus.UNDETERMINED.value, device=self.device),
                ),
            )

        return status


@dataclass
class GenerationOutput:
    """Results from sequence generation."""

    sequences: torch.Tensor  # shape: [batch_size, seq_len]
    satisfied: torch.Tensor  # shape: [batch_size], boolean
    impossible: torch.Tensor  # shape: [batch_size], boolean
    times: torch.Tensor  # shape: [batch_size, seq_len], time deltas


def get_task_labeler(
    task_config: TaskExtractorConfig, device: str = "cpu", batch_size: int = 1
) -> BatchAutoregressiveLabeler:
    """Create a BatchAutoregressiveLabeler from task config."""
    return BatchAutoregressiveLabeler(
        task_config=task_config,
        token_to_code_map=task_config.token_to_code_map,
        batch_size=batch_size,
        device=device,
    )


@dataclass
class TrajectoryBatch:
    """
    Initialize a batch of trajectories.

    Args:
        time (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing days after prediction
            time. Values must be monotonically increasing within each sequence.
        code (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing event codes
        mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicates valid measurements/codes
        numeric_value (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing numeric values
        numeric_value_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) indicating valid
            numeric values
        time_scale: scale of the time, by default it is 'Y' (years). Any numpy datetime units can be used,
            see https://numpy.org/doc/2.1/reference/arrays.datetime.html#datetime-units
    """

    time: torch.Tensor
    code: torch.Tensor
    mask: torch.Tensor
    numeric_value: torch.Tensor
    numeric_value_mask: torch.Tensor
    time_scale: str = "Y"

    def to_meds(self, prediction_time: list[datetime], subject_id: list[str | int]) -> pl.DataFrame:
        """Convert the trajectory batch to MEDS format.

        Args:
            prediction_time: List of prediction times for each trajectory in the batch
            subject_id: List of subject IDs for each trajectory in the batch

        Returns:
            pl.DataFrame: MEDS format DataFrame with columns:
                - time: Absolute timestamp of the event
                - code: The medical code
                - numeric_value: The numeric value associated with the code (if any)
                - subject_id: ID of the subject
                - prediction_time: The prediction time for this trajectory

        Example:
        >>> batch = TrajectoryBatch(
        ...     time=torch.tensor([[0, .5, 2], [0, 3, 5]]),
        ...     code=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ...     mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        ...     numeric_value=torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]),
        ...     numeric_value_mask=torch.tensor([[1, 1, 0], [1, 0, 0]])
        ... )
        >>> prediction_times = [datetime(2024, 1, 1), datetime(2024, 1, 1)]
        >>> subject_ids = [1, 2]
        >>> df = batch.to_meds(prediction_times, subject_ids)
        >>> df.sort("subject_id")
        shape: (5, 5)
        ┌─────────────────────┬──────┬───────────────┬────────────┬─────────────────────┐
        │ time                ┆ code ┆ numeric_value ┆ subject_id ┆ prediction_time     │
        │ ---                 ┆ ---  ┆ ---           ┆ ---        ┆ ---                 │
        │ datetime[ns]        ┆ i32  ┆ f32           ┆ i32        ┆ datetime[ns]        │
        ╞═════════════════════╪══════╪═══════════════╪════════════╪═════════════════════╡
        │ 2024-01-01 00:00:00 ┆ 1    ┆ 0.5           ┆ 1          ┆ 2024-01-01 00:00:00 │
        │ 2024-07-01 14:54:36 ┆ 2    ┆ 1.0           ┆ 1          ┆ 2024-01-01 00:00:00 │
        │ 2025-12-31 11:38:24 ┆ 3    ┆ NaN           ┆ 1          ┆ 2024-01-01 00:00:00 │
        │ 2024-01-01 00:00:00 ┆ 4    ┆ 2.0           ┆ 2          ┆ 2024-01-01 00:00:00 │
        │ 2026-12-31 17:27:36 ┆ 5    ┆ NaN           ┆ 2          ┆ 2024-01-01 00:00:00 │
        └─────────────────────┴──────┴───────────────┴────────────┴─────────────────────┘
        """
        if len(prediction_time) != len(subject_id) or len(prediction_time) != self.time.shape[0]:
            raise ValueError("Number of prediction times and subject IDs must match batch size")
        schema = {
            "time": pl.Datetime,
            "code": pl.Int32,
            "numeric_value": pl.Float32,
            "subject_id": pl.Int32,
            "prediction_time": pl.Datetime,
        }

        # Pre-filter masked data using torch operations for efficiency
        batch_indices, seq_indices = torch.where(self.mask)
        if len(batch_indices) == 0:
            return pl.DataFrame(schema=schema)

        # Gather only the valid data points using index tensors
        time_values = self.time[batch_indices, seq_indices]
        code_values = self.code[batch_indices, seq_indices]
        numeric_values = self.numeric_value[batch_indices, seq_indices]
        numeric_value_masks = self.numeric_value_mask[batch_indices, seq_indices]

        # Convert to numpy for faster processing
        time_array = time_values.numpy()
        code_array = code_values.numpy().astype(np.int32)
        numeric_value_array = numeric_values.numpy()
        numeric_value_mask_array = numeric_value_masks.numpy()
        batch_indices = batch_indices.numpy()

        # Create arrays for prediction times and subject IDs
        pred_times = np.array(prediction_time, dtype="datetime64[ns]")[batch_indices]
        subject_ids = np.array(subject_id)[batch_indices].astype(np.int32)

        # Parallel processing using numpy vectorization
        time_deltas = time_array * np.timedelta64(1, self.time_scale).astype("timedelta64[ns]")
        timestamps = pred_times + time_deltas

        # Create the final dictionary with only valid data
        data_dict = {
            "time": timestamps,
            "code": code_array,
            "numeric_value": np.where(
                numeric_value_mask_array, numeric_value_array.astype(np.float32), np.nan
            ),
            "subject_id": subject_ids,
            "prediction_time": pred_times,
        }

        # Create DataFrame directly from the efficient dictionary
        return pl.from_dict(data_dict, schema=schema)


class GenerationTracker:
    """Tracks generation progress and stopping criteria for token-by-token generation.

    Args:
        batch_size: Number of sequences being generated
        eos_tokens: List of tokens that should stop generation (from task config)
        budget: Optional GenerationBudget with length/time constraints
        device: Device for torch tensors

    Examples:
        >>> # Initialize tracker
        >>> tracker = GenerationTracker(
        ...     batch_size=2,
        ...     eos_tokens=[1, 2],  # death or discharge tokens
        ...     budget=GenerationBudget(max_seq_len=100, max_time=48.0)
        ... )
        >>>
        >>> # Track token by token
        >>> next_token = torch.tensor([3, 4])  # Some non-EOS tokens
        >>> tracker.update(next_token)
        >>> tracker.should_stop
        False
        >>> tracker.finished_sequences
        tensor([False, False])
        >>>
        >>> # Hit EOS token
        >>> next_token = torch.tensor([1, 3])  # First sequence hits death token
        >>> tracker.update(next_token)
        >>> tracker.should_stop
        False
        >>> tracker.finished_sequences
        tensor([ True, False])
        >>>
        >>> # Get masks for unfinished sequences
        >>> tracker.unfinished_mask
        tensor([False,  True])
    """

    def __init__(
        self,
        batch_size: int,
        eos_tokens: list[int],
        budget: GenerationBudget | None = None,
        device: str = "cpu",
    ):
        self.batch_size = batch_size
        self.eos_tokens = torch.tensor(eos_tokens, device=device) if eos_tokens else None
        self.budget = budget
        self.device = device

        # Initialize state
        self.num_generated = 0
        self.cumulative_time = torch.zeros(batch_size, device=device)
        self.finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def update(self, next_token: torch.Tensor, next_token_time: float | None = None) -> None:
        """Update tracker with next generated token(s).

        Args:
            next_token: Tensor of shape (batch_size) with next token IDs
            next_token_time: Optional time value for time budget tracking
        """
        # Update counts
        self.num_generated += 1

        # Check for EOS tokens
        if self.eos_tokens is not None:
            self.finished_sequences |= torch.isin(next_token, self.eos_tokens)

        # Check budgets if specified
        if self.budget:
            # Sequence length budget
            if self.budget.max_seq_len and self.num_generated >= self.budget.max_seq_len:
                self.finished_sequences.fill_(True)

            # Time budget
            if self.budget.max_time and next_token_time:
                self.cumulative_time += next_token_time
                self.finished_sequences |= self.cumulative_time >= self.budget.max_time

    @property
    def should_stop(self) -> bool:
        """Whether all sequences are finished."""
        return bool(self.finished_sequences.all().item())

    @property
    def unfinished_mask(self) -> torch.Tensor:
        """Boolean mask of shape (batch_size) indicating unfinished sequences."""
        return ~self.finished_sequences


@dataclass
class ZeroShotTaskConfig:
    """
    Unified configuration for zero-shot task extraction that combines generation and labeling logic.

    Args:
        task_config: The TaskExtractorConfig defining windows and predicates
        token_to_code_map: Dictionary mapping token IDs to predicate codes

    Examples:
        >>> # Create a simple ICU mortality task config
        >>> import yaml
        >>> icu_yaml = '''
        ... predicates:
        ...   icu_admission:
        ...     code: "event_type//ICU_ADMISSION"
        ...   death:
        ...     code: "event_type//DEATH"
        ...   discharge:
        ...     code: "event_type//DISCHARGE"
        ...   death_or_discharge:
        ...     expr: "or(death, discharge)"
        ... trigger: "icu_admission"
        ... windows:
        ...   observation:
        ...     start: null
        ...     end: "trigger + 24h"
        ...     start_inclusive: true
        ...     end_inclusive: true
        ...     has:
        ...       "_ANY_EVENT": "(1, None)"
        ...     index_timestamp: "end"
        ...   outcome:
        ...     start: "observation.end"
        ...     end: "start -> death_or_discharge"
        ...     start_inclusive: false
        ...     end_inclusive: true
        ...     has: {}
        ...     label: "death"
        ... '''
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as temp_dir:
        ...     yaml_fp = Path(temp_dir) / "icu.yaml"
        ...     _ = yaml_fp.write_text(icu_yaml)
        ...     task_config = TaskExtractorConfig.load(yaml_fp)
        >>> token_map = {0: "event_type//ICU_ADMISSION",
        ...              1: "event_type//DEATH",
        ...              2: "event_type//DISCHARGE"}
        >>> config = ZeroShotTaskConfig(task_config, token_map)
        >>> # Get generation budget
        >>> budget = config.get_generation_budget()
        >>> assert budget.max_seq_len is None
        >>> assert budget.max_time is None
        >>> # Get EOS tokens
        >>> config.get_eos_tokens() == [1, 2]  # death_or_discharge tokens
        True
        >>> # Get task labeler
        >>> labeler = config.get_task_labeler()
        >>> isinstance(labeler, BatchAutoregressiveLabeler)
        True
    """

    task_config: TaskExtractorConfig
    token_to_code_map: dict[int, str]
    max_seq_len: int | None = None

    def _get_tokens_for_predicate(self, predicate_name: str) -> list[int]:
        """Get token IDs that correspond to a predicate."""
        predicate = self.task_config.predicates[predicate_name]
        if hasattr(predicate, "code"):
            return [k for k, v in self.token_to_code_map.items() if v == predicate.code]
        elif hasattr(predicate, "expr"):
            # Handle derived predicates
            if predicate.expr.startswith("or("):
                predicates = [p.strip() for p in predicate.expr[3:-1].split(",")]
                return [t for p in predicates for t in self._get_tokens_for_predicate(p)]
            elif predicate.expr.startswith("and("):
                # For AND predicates in generation, we need all tokens
                predicates = [p.strip() for p in predicate.expr[4:-1].split(",")]
                return [t for p in predicates for t in self._get_tokens_for_predicate(p)]
        return []

    def get_generation_budget(self) -> GenerationBudget:
        """Extract generation budget from window configuration."""
        label_window = self.task_config.windows[self.task_config.label_window]

        # If the window has a time constraint, use time budget
        if "+" in str(label_window.end):
            time_str = label_window.end.split("+")[1].strip()
            # Convert time string to hours (simplified)
            if "h" in time_str:
                hours = float(time_str.replace("h", "")) / 24 / 365.25
            elif "d" in time_str:
                hours = float(time_str.replace("d", "")) / 365.25
            else:
                task_config_str = ""
                for branch, stem, node in yield_tree(self.task_config.window_tree):
                    task_config_str += f"{branch}{stem}{node.node_name}"
                raise ValueError(
                    f"Unsupported time symbol in the task config: {time_str}. "
                    f"View the offending task_config below:{task_config_str}"
                )
            return GenerationBudget(max_seq_len=self.max_seq_len, max_time=hours)
        else:
            # If the window ends with a predicate, use predicate to end generation
            return GenerationBudget(max_seq_len=self.max_seq_len)  # fallback

    def get_eos_tokens(self) -> list[int]:
        """Get tokens that should stop generation."""
        label_window = self.task_config.windows[self.task_config.label_window]

        # If window ends with a predicate, use those tokens
        if "->" in str(label_window.end):
            pred_name = label_window.end.split("->")[1].strip()
            return self._get_tokens_for_predicate(pred_name)

        return []

    def get_task_labeler(self, batch_size: int = 1, device: str = "cpu") -> BatchAutoregressiveLabeler:
        """Get an autoregressive labeler for this task."""
        return BatchAutoregressiveLabeler(
            task_config=self.task_config,
            token_to_code_map=self.token_to_code_map,
            batch_size=batch_size,
            device=device,
        )


def generate(
    model,
    prompts: torch.Tensor,
    zs_task_config: ZeroShotTaskConfig,
    end_time_delta: float,  # Time between end_time and prediction_time
) -> GenerationOutput:
    """
    Generate sequences using both budget and task constraints.

    Args:
        model: Model with generate_next_token(prompts, temperature) method
        prompts: Starting token sequences [batch_size, prompt_len]
        zs_task_config: Zero-Shot Task configuration
        end_time_delta: Time between end_time and prediction_time
        budget: Optional generation budget constraints
        temperature: Sampling temperature
        get_next_token_time: Optional function to get time deltas for tokens
    """
    batch_size = prompts.shape[0]
    device = prompts.device

    budget = zs_task_config.get_generation_budget()

    # Initialize trackers
    generation_tracker = GenerationTracker(
        batch_size=batch_size,
        eos_tokens=None,  # TODO: EOS tokens are now conditions in the TaskConfig, use that
        budget=budget,
        device=device,
    )

    # Initialize time reference
    time_ref = end_time_delta

    # Create task labeler
    labeler = BatchAutoregressiveLabeler(
        task_config=zs_task_config.task_config,
        token_to_code_map=zs_task_config.token_to_code_map,
        batch_size=batch_size,
        device=device,
    )

    # Initialize output tensors
    current_output = prompts.clone()
    time_deltas = torch.zeros(batch_size, 1, device=device)
    satisfied = torch.zeros(batch_size, dtype=torch.bool, device=device)
    impossible = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Generate until all sequences are done
    while not generation_tracker.should_stop:
        # Generate next tokens for unfinished sequences
        active_mask = generation_tracker.unfinished_mask

        if active_mask.any():
            # Get next tokens from model
            next_codes, next_times, next_numeric_values = model.generate_next_token(
                current_output[active_mask],
            )
            time_ref += next_times

            # Update generation tracker
            generation_tracker.update(next_codes, time_ref)

            # Update task labeler
            task_status = labeler.update(next_codes, time_ref)

            # Update sequence statuses
            newly_satisfied = task_status == ConstraintStatus.SATISFIED.value
            newly_impossible = task_status == ConstraintStatus.IMPOSSIBLE.value

            satisfied[active_mask] |= newly_satisfied
            impossible[active_mask] |= newly_impossible

            # Mark sequences as finished if task status determined
            generation_tracker.finished_sequences[active_mask] |= newly_satisfied | newly_impossible

            # Append tokens
            new_output = torch.cat(
                [current_output, torch.zeros(batch_size, 1, dtype=torch.long, device=device)], dim=1
            )
            new_output[active_mask, -1] = next_codes
            current_output = new_output
            # TODO remove
            logger.info("satisfied:   {newly_satisfied}")
            logger.info(f"impossible: {newly_impossible}")
            logger.info(generation_tracker.finished_sequences[active_mask])

    return GenerationOutput(
        sequences=current_output, satisfied=satisfied, impossible=impossible, times=time_deltas
    )
