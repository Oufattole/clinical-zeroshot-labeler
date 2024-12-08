import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import polars as pl
import torch

from czsl.config import TaskExtractorConfig
from czsl.utils import parse_timedelta


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
        tensor([True, False])
        >>>
        >>> # Get masks for unfinished sequences
        >>> tracker.unfinished_mask
        tensor([False, True])
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
        return self.finished_sequences.all()

    @property
    def unfinished_mask(self) -> torch.Tensor:
        """Boolean mask of shape (batch_size) indicating unfinished sequences."""
        return ~self.finished_sequences


def generate_sequences(
    model,
    prompts: torch.Tensor,
    task_config: "ZeroShotTaskConfig",
    temperature: float = 1.0,
    budget: GenerationBudget | None = None,
    get_next_token_time: callable | None = None,
) -> torch.Tensor:
    """Generate sequences using task configuration for stopping criteria.

    Args:
        model: Model with a generate_next_token method
        prompts: Initial token sequences
        task_config: Task configuration
        temperature: Sampling temperature
        budget: Optional generation constraints
        get_next_token_time: Optional function to get time for tokens

    Returns:
        Generated sequences
    """
    batch_size = prompts.shape[0]
    device = prompts.device

    # Initialize generation tracker
    tracker = GenerationTracker(
        batch_size=batch_size, eos_tokens=task_config.get_eos_tokens(), budget=budget, device=device
    )

    # Initialize output tensors
    output = prompts.clone()

    # Generate tokens until stopping criteria met
    while not tracker.should_stop:
        # Get next token for unfinished sequences
        next_token = model.generate_next_token(output[tracker.unfinished_mask], temperature=temperature)

        # Update time tracking if needed
        next_token_time = None
        if get_next_token_time and budget and budget.max_time:
            next_token_time = get_next_token_time(next_token)

        # Update tracker
        tracker.update(next_token, next_token_time)

        # Append tokens
        if tracker.unfinished_mask.any():
            new_output = torch.cat([output, torch.zeros_like(next_token).unsqueeze(1)], dim=1)
            new_output[tracker.unfinished_mask, -1] = next_token
            output = new_output

    return output


@dataclass
class TaskLabeler:
    """
    Initialize a time to event labeler.

    Args:
        target_codes: List of codes we're looking for in the sequence
        max_time: The maximum time window to look for the event
        min_time: Optional minimum time that must pass before events are considered valid
        numeric_value_min: Optional minimum value for numeric criteria
        numeric_value_max: Optional maximum value for numeric criteria
        include_min_time: Whether to include events exactly at min_time
        include_max_time: Whether to include events exactly at max_time
        require_max_time: Whether to return unknown if max_time is not reached

    Examples:
        >>> # Case 1: Testing unknown_pred logic
        >>> time = torch.tensor([
        ...     [0.0, 2.0, 5.0, 12.0],  # Has event at day 5
        ...     [0.0, 2.0, 3.0, 20.0],  # No event, but seen full window
        ...     [0.0, 2.0, 4.0, 10.0],  # No event, haven't seen full window yet
        ... ])
        >>> code = torch.tensor([
        ...     [1, 2, 100, 3],    # Has target code
        ...     [1, 2, 3, 4],      # No target codes, full window seen
        ...     [1, 2, 3, 4],      # No target codes, partial window
        ... ])
        >>> mask = torch.ones_like(time, dtype=torch.bool)
        >>> numeric_value = torch.ones_like(time)
        >>> numeric_value_mask = torch.ones_like(time, dtype=torch.bool)
        >>> batch = TrajectoryBatch(time, code, mask, numeric_value, numeric_value_mask)
        >>> labeler = TaskLabeler(target_codes=[100], max_time=15.0, require_max_time=True)
        >>> labels, unknown = labeler(batch)
        >>> print(labels.tolist())  # Only first sequence has event
        [[1.0], [0.0], [0.0]]
        >>> print(unknown.tolist())  # Only third sequence is unknown (hasn't seen full window)
        [False, False, True]

        >>> # Case 2: Complex case with multiple codes and criteria
        >>> time = torch.tensor([
        ...     [0.0, 3.0, 5.0, 35.0],   # Has valid event at day 5
        ...     [0.0, 2.0, 6.0, 35.0],   # No valid events, full window seen
        ...     [0.0, 2.0, 4.0, 10.0],   # No valid events, partial window
        ...     [0.0, 35.0, 40.0, 45.0], # Events outside time window
        ... ])
        >>> code = torch.tensor([
        ...     [1, 2, 100, 101],    # Has first target code
        ...     [100, 2, 3, 4],      # Has early target code
        ...     [1, 2, 3, 4],        # No target codes
        ...     [1, 100, 101, 2],    # Target codes but too late
        ... ])
        >>> mask = torch.ones_like(time, dtype=torch.bool)
        >>> numeric_value = torch.tensor([
        ...     [1.0, 1.0, 7.5, 8.0],  # Valid values
        ...     [6.0, 1.0, 4.5, 4.0],  # Valid values
        ...     [1.0, 1.0, 4.0, 4.0],  # Values don't matter (no target codes)
        ...     [1.0, 7.0, 8.0, 1.0],  # Valid values but events too late
        ... ])
        >>> numeric_value_mask = torch.ones_like(time, dtype=torch.bool)
        >>> batch = TrajectoryBatch(time, code, mask, numeric_value, numeric_value_mask)
        >>> labeler = TaskLabeler(target_codes=[100, 101], max_time=30.0, min_time=5.0,
        ...                      numeric_value_min=5.0, numeric_value_max=10.0, require_max_time=True)
        >>> labels, unknown = labeler(batch)
        >>> print(labels.tolist())  # Only first sequence has valid event
        [[1.0], [0.0], [0.0], [0.0]]
        >>> print(unknown.tolist())  # Only third sequence is unknown
        [False, False, True, False]

        >>> # Case 3: No Max Time Requirement
        >>> labeler = TaskLabeler(target_codes=[100, 101], max_time=30.0, min_time=5.0,
        ...                      numeric_value_min=5.0, numeric_value_max=10.0)
        >>> labels, unknown = labeler(batch)
        >>> print(labels.tolist())  # Only first sequence has valid event
        [[1.0], [0.0], [0.0], [0.0]]
        >>> print(unknown.tolist())  # Only third sequence is unknown
        [False, False, False, False]
    """

    target_codes: list[int]
    max_time: float
    min_time: float = 0.0
    numeric_value_min: float | None = None
    numeric_value_max: float | None = None
    include_min_time: bool = True
    include_max_time: bool = True
    require_max_time: bool = False

    def __call__(self, trajectory_batch: TrajectoryBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Label sequences for time to event prediction.

        Args:
            trajectory (Trajectory): The trajectory data

        Returns:
            tuple of:
                pred_labels: Tensor of shape (batch_size, 1) containing binary labels
                unknown_pred: Tensor of shape (batch_size,) indicating if no prediction could be made
                    Only True if no event found AND max time < self.max_time and self.require_max_time
                    is true
        """
        time = trajectory_batch.time
        code = trajectory_batch.code
        mask = trajectory_batch.mask
        numeric_value = trajectory_batch.numeric_value
        numeric_value_mask = trajectory_batch.numeric_value_mask

        # Find where any target code appears and is valid
        is_target_code = torch.zeros_like(code, dtype=torch.bool)
        for target_code in self.target_codes:
            is_target_code = is_target_code | (code == target_code)
        is_target_code = is_target_code & mask

        # Apply time window constraint
        in_time_window = time == torch.clamp(time, self.min_time, self.max_time)
        if not self.include_min_time:
            in_time_window = in_time_window & (time != self.min_time)
        if not self.include_max_time:
            in_time_window = in_time_window & (time != self.max_time)
        valid_events = is_target_code & in_time_window

        # Apply numeric constraints if specified
        if self.numeric_value_min is not None or self.numeric_value_max is not None:
            numeric_criteria = torch.ones_like(valid_events, dtype=torch.bool)

            if self.numeric_value_min is not None:
                numeric_criteria = numeric_criteria & (numeric_value >= self.numeric_value_min)

            if self.numeric_value_max is not None:
                numeric_criteria = numeric_criteria & (numeric_value <= self.numeric_value_max)

            valid_events = valid_events & numeric_criteria & numeric_value_mask

        # Find first valid event for each sequence
        has_event = valid_events.any(dim=1)

        # Create prediction labels
        pred_labels = has_event.unsqueeze(1).float()

        # Mark sequences as unknown only if:
        # 1. No valid events found AND
        # 2. Haven't seen the full time window yet (max time < time_length)
        max_times = torch.amax(time * mask.float(), dim=1)  # Use mask to ignore invalid times
        window_incomplete = max_times < self.max_time
        if self.require_max_time:
            unknown_pred = ~has_event & window_incomplete
        else:
            unknown_pred = torch.zeros_like(has_event, dtype=torch.bool)

        return pred_labels, unknown_pred


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
        >>> budget.budget_type == BudgetType.EOS_ONLY
        True
        >>> # Get EOS tokens
        >>> config.get_eos_tokens() == [1, 2]  # death_or_discharge tokens
        True
        >>> # Get task labeler
        >>> labeler = config.get_task_labeler()
        >>> labeler.target_codes == [1]  # death token
        True
    """

    task_config: TaskExtractorConfig
    token_to_code_map: dict[int, str]

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

        # If the window ends with a predicate, use EOS-only budget
        if "->" in str(label_window.end):
            return GenerationBudget.from_eos_only()

        # If the window has a time constraint, use time budget
        if "+" in str(label_window.end):
            time_str = label_window.end.split("+")[1].strip()
            # Convert time string to hours (simplified)
            if "h" in time_str:
                hours = float(time_str.replace("h", ""))
            elif "d" in time_str:
                hours = float(time_str.replace("d", "")) * 24
            return GenerationBudget.from_time_len(hours)

        return GenerationBudget.from_seq_len(1000)  # fallback

    def get_eos_tokens(self) -> list[int]:
        """Get tokens that should stop generation."""
        label_window = self.task_config.windows[self.task_config.label_window]

        # If window ends with a predicate, use those tokens
        if "->" in str(label_window.end):
            pred_name = label_window.end.split("->")[1].strip()
            return self._get_tokens_for_predicate(pred_name)

        return []

    def get_task_labeler(self) -> TaskLabeler:
        """Create a TaskLabeler from the config."""
        label_window = self.task_config.windows[self.task_config.label_window]

        # Get target tokens for labeling
        target_codes = self._get_tokens_for_predicate(label_window.label)

        # Extract time constraints
        max_time = None
        min_time = 0.0
        if label_window.end and "+" in label_window.end:
            time_str = label_window.end.split("+")[1].strip()
            if "h" in time_str:
                max_time = float(time_str.replace("h", ""))
            elif "d" in time_str:
                max_time = float(time_str.replace("d", "")) * 24

        return TaskLabeler(
            target_codes=target_codes,
            max_time=max_time,
            min_time=min_time,
            include_min_time=label_window.start_inclusive,
            include_max_time=label_window.end_inclusive,
        )


def create_zero_shot_task(yaml_path: str, token_to_code_map: dict[int, str]) -> ZeroShotTaskConfig:
    """Create a ZeroShotTaskConfig from a YAML file and token mapping.

    Args:
        yaml_path: Path to the task YAML file
        token_to_code_map: Dictionary mapping token IDs to predicate codes

    Returns:
        ZeroShotTaskConfig configured for the task
    """
    task_config = TaskExtractorConfig.load(yaml_path)
    return ZeroShotTaskConfig(task_config, token_to_code_map)


class ConstraintStatus(Enum):
    """Status of a constraint evaluation."""

    UNDETERMINED = 0
    SATISFIED = 1
    IMPOSSIBLE = 2


class BatchConstraintState:
    """Base class for tracking constraint state across a batch."""

    batch_size: int

    def update(self, tokens: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
        """
        Update state with new tokens and return status for each sequence.

        Args:
            tokens: Tensor of shape (batch_size,) with token IDs
            times: Optional tensor of shape (batch_size,) with cumulative times

        Returns:
            Tensor of shape (batch_size,) with ConstraintStatus values
        """
        raise NotImplementedError


@dataclass
class BatchTokenMatchState(BatchConstraintState):
    """Track token occurrences across batch."""

    target_tokens: set[int] = None
    min_count: int | None = None
    max_count: int | None = None
    counts: torch.Tensor | None = None
    device: str = "cpu"

    def __post_init__(self):
        self.counts = torch.zeros(self.batch_size, device=self.device)
        self.target_tokens_tensor = torch.tensor(list(self.target_tokens), device=self.device)

    def update(self, tokens: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
        # Update counts where tokens match targets
        self.counts += torch.isin(tokens, self.target_tokens_tensor).float()

        # Initialize status as UNDETERMINED
        status = torch.full_like(tokens, ConstraintStatus.UNDETERMINED.value)

        # Update status based on counts
        if self.max_count is not None:
            status = torch.where(
                self.counts > self.max_count,
                torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=self.device),
                status,
            )

        if self.min_count is not None:
            status = torch.where(
                (self.counts >= self.min_count) & (status == ConstraintStatus.UNDETERMINED.value),
                torch.tensor(ConstraintStatus.SATISFIED.value, device=self.device),
                status,
            )

        return status


@dataclass
class BatchTimeWindowState(BatchConstraintState):
    """Track time windows across batch."""

    min_time: float | None = None
    max_time: float | None = None
    include_min: bool = True
    include_max: bool = True
    current_times: torch.Tensor | None = None
    device: str = "cpu"

    def __post_init__(self):
        self.current_times = torch.zeros(self.batch_size, device=self.device)

    def update(self, tokens: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
        if times is None:
            return torch.full_like(tokens, ConstraintStatus.UNDETERMINED.value)

        self.current_times = times
        status = torch.full_like(tokens, ConstraintStatus.UNDETERMINED.value)

        if self.max_time is not None:
            if self.include_max:
                status = torch.where(
                    times > self.max_time,
                    torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=self.device),
                    status,
                )
            else:
                status = torch.where(
                    times >= self.max_time,
                    torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=self.device),
                    status,
                )

        if self.min_time is not None:
            mask = status == ConstraintStatus.UNDETERMINED.value
            if self.include_min:
                status = torch.where(
                    mask & (times >= self.min_time),
                    torch.tensor(ConstraintStatus.SATISFIED.value, device=self.device),
                    status,
                )
            else:
                status = torch.where(
                    mask & (times > self.min_time),
                    torch.tensor(ConstraintStatus.SATISFIED.value, device=self.device),
                    status,
                )

        return status


@dataclass
class BatchCompositeState(BatchConstraintState):
    """Combine multiple constraints with AND/OR logic across batch."""

    states: list[BatchConstraintState]
    operator: str  # 'and' or 'or'
    device: str = "cpu"

    def update(self, tokens: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
        results = torch.stack([state.update(tokens, times) for state in self.states])

        if self.operator == "and":
            # IMPOSSIBLE if any state is IMPOSSIBLE
            status = torch.full_like(tokens, ConstraintStatus.UNDETERMINED.value)
            status = torch.where(
                (results == ConstraintStatus.IMPOSSIBLE.value).any(dim=0),
                torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=self.device),
                status,
            )
            # SATISFIED if all states are SATISFIED
            status = torch.where(
                (results == ConstraintStatus.SATISFIED.value).all(dim=0),
                torch.tensor(ConstraintStatus.SATISFIED.value, device=self.device),
                status,
            )
        else:  # or
            # SATISFIED if any state is SATISFIED
            status = torch.full_like(tokens, ConstraintStatus.UNDETERMINED.value)
            status = torch.where(
                (results == ConstraintStatus.SATISFIED.value).any(dim=0),
                torch.tensor(ConstraintStatus.SATISFIED.value, device=self.device),
                status,
            )
            # IMPOSSIBLE if all states are IMPOSSIBLE
            status = torch.where(
                (results == ConstraintStatus.IMPOSSIBLE.value).all(dim=0),
                torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=self.device),
                status,
            )

        return status


@dataclass
class BatchWindowState(BatchConstraintState):
    """Track a complete window with time bounds and constraints across batch."""

    time_bounds: BatchTimeWindowState
    constraints: list[BatchConstraintState]
    device: str = "cpu"

    def update(self, tokens: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
        time_status = self.time_bounds.update(tokens, times)

        # Initialize with time status
        status = time_status.clone()

        # Where time conditions are met, check other constraints
        time_mask = time_status == ConstraintStatus.SATISFIED.value
        if time_mask.any():
            constraint_results = torch.stack(
                [
                    c.update(tokens[time_mask], times[time_mask] if times is not None else None)
                    for c in self.constraints
                ]
            )

            # Update status where time conditions are met
            impossible_mask = (constraint_results == ConstraintStatus.IMPOSSIBLE.value).any(dim=0)
            satisfied_mask = (constraint_results == ConstraintStatus.SATISFIED.value).all(dim=0)

            status[time_mask] = torch.where(
                impossible_mask,
                torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=self.device),
                torch.where(
                    satisfied_mask,
                    torch.tensor(ConstraintStatus.SATISFIED.value, device=self.device),
                    torch.tensor(ConstraintStatus.UNDETERMINED.value, device=self.device),
                ),
            )

        return status


class BatchAutoregressiveLabeler:
    """
    Batch implementation of autoregressive task labeling.

    Examples:
        >>> # Setup for a batch of 3 sequences
        >>> batch_size = 3
        >>> device = "cpu"
        >>>
        >>> # Create window states
        >>> observation = BatchWindowState(
        ...     batch_size=batch_size,
        ...     device=device,
        ...     time_bounds=BatchTimeWindowState(
        ...         batch_size=batch_size,
        ...         device=device,
        ...         max_time=24.0
        ...     ),
        ...     constraints=[
        ...         BatchTokenMatchState(
        ...             batch_size=batch_size,
        ...             device=device,
        ...             target_tokens={1, 2},
        ...             max_count=0
        ...         )
        ...     ]
        ... )
        >>>
        >>> labeler = BatchAutoregressiveLabeler([observation])
        >>>
        >>> # Test batch update
        >>> tokens = torch.tensor([0, 1, 2], device=device)  # One valid, two invalid
        >>> times = torch.tensor([12.0, 12.0, 12.0], device=device)
        >>> status = labeler.update(tokens, times)
        >>> # First sequence still valid, others failed
        >>> (status == torch.tensor([
        ...     ConstraintStatus.UNDETERMINED.value,
        ...     ConstraintStatus.IMPOSSIBLE.value,
        ...     ConstraintStatus.IMPOSSIBLE.value
        ... ])).all()
        True
    """

    def __init__(self, windows: list[BatchWindowState]):
        self.windows = windows

    def update(self, tokens: torch.Tensor, times: torch.Tensor | None = None) -> torch.Tensor:
        """
        Update with new tokens and return status for each sequence.

        Args:
            tokens: Tensor of shape (batch_size,) with token IDs
            times: Optional tensor of shape (batch_size,) with cumulative times

        Returns:
            Tensor of shape (batch_size,) with ConstraintStatus values
        """
        results = torch.stack([window.update(tokens, times) for window in self.windows])

        # Initialize status
        status = torch.full_like(tokens, ConstraintStatus.UNDETERMINED.value)

        # Task fails if any window is impossible
        status = torch.where(
            (results == ConstraintStatus.IMPOSSIBLE.value).any(dim=0),
            torch.tensor(ConstraintStatus.IMPOSSIBLE.value, device=status.device),
            status,
        )

        # Task succeeds if all windows are satisfied
        status = torch.where(
            (results == ConstraintStatus.SATISFIED.value).all(dim=0),
            torch.tensor(ConstraintStatus.SATISFIED.value, device=status.device),
            status,
        )

        return status

    @classmethod
    def from_task_config(
        cls,
        task_config: "TaskExtractorConfig",
        token_to_code_map: dict[int, str],
        batch_size: int,
        device: str = "cpu",
    ) -> "BatchAutoregressiveLabeler":
        """Create BatchAutoregressiveLabeler from ACES task config.

        Args:
            task_config: ACES task configuration
            token_to_code_map: Mapping from token IDs to predicate codes
            batch_size: Number of sequences in batch
            device: Torch device

        Returns:
            Configured BatchAutoregressiveLabeler

        Examples:
            >>> # Create from config
            >>> batch_size = 2
            >>> device = "cpu"
            >>> labeler = BatchAutoregressiveLabeler.from_task_config(
            ...     task_config=task_config,
            ...     token_to_code_map={
            ...         0: "PAD",
            ...         1: "ICU_ADMISSION//MEDICAL",
            ...         2: "HOSPITAL_DISCHARGE//MEDICAL",
            ...         3: "MEDS_DEATH"
            ...     },
            ...     batch_size=batch_size,
            ...     device=device
            ... )
        """

        def create_predicate_state(predicate_name: str) -> BatchConstraintState:
            """Create a batched constraint state for a predicate."""
            predicate = task_config.predicates[predicate_name]
            if hasattr(predicate, "code"):
                # Plain predicate
                # Handle regex and exact matches
                if isinstance(predicate.code, dict) and "regex" in predicate.code:
                    # For regex, compile pattern and match against values
                    pattern = re.compile(predicate.code["regex"])
                    tokens = {k for k, v in token_to_code_map.items() if pattern.match(v)}
                else:
                    # For exact match
                    tokens = {k for k, v in token_to_code_map.items() if v == predicate.code}
                return BatchTokenMatchState(batch_size=batch_size, device=device, target_tokens=tokens)
            else:
                # Derived predicate
                if predicate.expr.startswith("and("):
                    preds = [p.strip() for p in predicate.expr[4:-1].split(",")]
                    return BatchCompositeState(
                        batch_size=batch_size,
                        device=device,
                        states=[create_predicate_state(p) for p in preds],
                        operator="and",
                    )
                else:  # or
                    preds = [p.strip() for p in predicate.expr[3:-1].split(",")]
                    return BatchCompositeState(
                        batch_size=batch_size,
                        device=device,
                        states=[create_predicate_state(p) for p in preds],
                        operator="or",
                    )

        # Create window states
        window_states = []
        for window_name, window_config in task_config.windows.items():
            # Create time bounds
            time_bounds = BatchTimeWindowState(
                batch_size=batch_size,
                device=device,
                min_time=0.0 if window_config.start is None else parse_timedelta(window_config.start),
                max_time=None if window_config.end is None else parse_timedelta(window_config.end),
                include_min=window_config.start_inclusive,
                include_max=window_config.end_inclusive,
            )

            # Create constraints for each predicate
            constraints = []
            for pred_name, (min_count, max_count) in window_config.has.items():
                pred_state = create_predicate_state(pred_name)
                if isinstance(pred_state, BatchTokenMatchState):
                    pred_state.min_count = min_count
                    pred_state.max_count = max_count
                constraints.append(pred_state)

            window_states.append(
                BatchWindowState(
                    batch_size=batch_size, device=device, time_bounds=time_bounds, constraints=constraints
                )
            )

        return cls(windows=window_states)


@dataclass
class GenerationOutput:
    """Results from sequence generation."""

    sequences: torch.Tensor  # shape: [batch_size, seq_len]
    satisfied: torch.Tensor  # shape: [batch_size], boolean
    impossible: torch.Tensor  # shape: [batch_size], boolean


def generate(
    model,
    prompts: torch.Tensor,
    task_config: "ZeroShotTaskConfig",
    budget: Optional["GenerationBudget"] = None,
    temperature: float = 1.0,
    get_next_token_time: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> GenerationOutput:
    """
    Generate sequences using both budget and task constraints.

    Args:
        model: Model with generate_next_token(prompts, temperature) method
        prompts: Starting token sequences [batch_size, prompt_len]
        task_config: Task configuration
        budget: Optional generation budget constraints
        temperature: Sampling temperature
        get_next_token_time: Optional function to get time deltas for tokens

    Returns:
        GenerationOutput containing sequences and their final statuses
    """
    batch_size = prompts.shape[0]
    device = prompts.device

    # Initialize trackers
    generation_tracker = GenerationTracker(
        batch_size=batch_size, eos_tokens=task_config.get_eos_tokens(), budget=budget, device=device
    )

    # Create task labeler using factory method
    labeler = BatchAutoregressiveLabeler.from_task_config(
        task_config=task_config.task_config,
        token_to_code_map=task_config.token_to_code_map,
        batch_size=batch_size,
        device=device,
    )

    # Initialize output tensors
    current_output = prompts.clone()
    satisfied = torch.zeros(batch_size, dtype=torch.bool, device=device)
    impossible = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Generate until all sequences are done
    while not generation_tracker.should_stop:
        # Generate next tokens for unfinished sequences
        active_mask = generation_tracker.unfinished_mask

        if active_mask.any():
            # Get next tokens from model
            next_tokens = model.generate_next_token(current_output[active_mask], temperature=temperature)

            # Get time if needed
            next_token_time = None
            if get_next_token_time is not None:
                next_token_time = get_next_token_time(next_tokens)

            # Update generation tracker
            generation_tracker.update(next_tokens, next_token_time)

            # Update task labeler
            task_status = labeler.update(next_tokens, generation_tracker.cumulative_time[active_mask])

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
            new_output[active_mask, -1] = next_tokens
            current_output = new_output

    return GenerationOutput(sequences=current_output, satisfied=satisfied, impossible=impossible)
