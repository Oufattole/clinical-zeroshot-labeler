from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import numpy as np
import polars as pl
import torch

from czsl.config import TaskExtractorConfig


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


class BudgetType(StrEnum):
    """Type of generation budget."""

    SEQUENCE_LENGTH = "sequence_length"
    TIME = "time"
    EOS_ONLY = "eos_only"


@dataclass
class GenerationBudget:
    """A class to handle generation budgets with mutually exclusive constraints.

    Can be one of:
    - Sequence length budget (max tokens to generate)
    - Time length budget (minimum time to generate)
    - EOS-only budget (generate until EOS token, tracking time optionally)

    Examples:
        >>> # Create from sequence length
        >>> budget_seq = GenerationBudget.from_seq_len(100)
        >>> budget_seq.budget_type
        <BudgetType.SEQUENCE_LENGTH: 'sequence_length'>
        >>> budget_seq.value
        100

        >>> # Create from time length
        >>> budget_time = GenerationBudget.from_time_len(60)
        >>> budget_time.budget_type
        <BudgetType.TIME: 'time'>
        >>> budget_time.value
        60

        >>> # Create budget that only stops on EOS
        >>> budget_eos = GenerationBudget.from_eos_only()
        >>> budget_eos.budget_type
        <BudgetType.EOS_ONLY: 'eos_only'>
        >>> budget_eos.value is None
        True
    """

    budget_type: BudgetType
    value: int | float | None = None

    @classmethod
    def from_seq_len(cls, value: int) -> "GenerationBudget":
        """Create a GenerationBudget from a maximum sequence length."""
        return cls(budget_type=BudgetType.SEQUENCE_LENGTH, value=value)

    @classmethod
    def from_time_len(cls, value: int) -> "GenerationBudget":
        """Create a GenerationBudget from a minimum time length."""
        return cls(budget_type=BudgetType.TIME, value=value)

    @classmethod
    def from_eos_only(cls) -> "GenerationBudget":
        """Create a GenerationBudget that only stops on EOS tokens."""
        return cls(budget_type=BudgetType.EOS_ONLY)


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
