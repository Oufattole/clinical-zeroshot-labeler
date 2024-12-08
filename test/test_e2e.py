import tempfile

import pytest
import torch

from czsl.labeler import create_zero_shot_task, generate


class DummyModel:
    """Dummy model that returns predefined sequence patterns."""

    def __init__(self, sequences: list[list[tuple[int, float]]]):
        """
        Args:
            sequences: List of [(token, time), ...] for each sequence in batch
        """
        self.sequences = sequences
        self.current_positions = [0] * len(sequences)

    def generate_next_token(self, prompts: torch.Tensor) -> torch.Tensor:
        """Return next token for each sequence."""
        tokens = []
        for i, seq in enumerate(self.sequences):
            if self.current_positions[i] < len(seq):
                tokens.append(seq[self.current_positions[i]][0])
                self.current_positions[i] += 1
            else:
                tokens.append(0)  # pad token
        return torch.tensor(tokens, device=prompts.device)

    def get_next_token_time(self, token: torch.Tensor) -> torch.Tensor:
        """Return time for each token."""
        times = []
        for i, seq in enumerate(self.sequences):
            pos = self.current_positions[i] - 1  # already incremented
            if pos < len(seq):
                times.append(seq[pos][1])
            else:
                times.append(0.0)
        return torch.tensor(times, device=token.device)


@pytest.fixture
def task_config_yaml():
    return """
    predicates:
      hospital_discharge:
        code: { regex: "^HOSPITAL_DISCHARGE//.*" }
      icu_admission:
        code: { regex: "^ICU_ADMISSION//.*" }
      icu_discharge:
        code: { regex: "^ICU_DISCHARGE//.*" }
      death:
        code: MEDS_DEATH
      discharge_or_death:
        expr: or(icu_discharge, death, hospital_discharge)

    trigger: icu_admission

    windows:
      input:
        start: null
        end: trigger + 24h
        start_inclusive: True
        end_inclusive: True
        index_timestamp: end
      gap:
        start: trigger
        end: start + 48h
        start_inclusive: False
        end_inclusive: True
        has:
          icu_admission: (None, 0)
          discharge_or_death: (None, 0)
      target:
        start: gap.end
        end: start -> discharge_or_death
        start_inclusive: False
        end_inclusive: True
        label: death
    """


@pytest.fixture
def token_map():
    return {
        0: "PAD",
        1: "ICU_ADMISSION//MEDICAL",
        2: "ICU_DISCHARGE//MEDICAL",
        3: "HOSPITAL_DISCHARGE//MEDICAL",
        4: "MEDS_DEATH",
        5: "OTHER_EVENT",
    }


@pytest.fixture
def successful_death_sequence():
    """Patient admitted to ICU, survives gap period, then dies."""
    return [
        (1, 0.0),  # ICU admission at t=0
        (5, 20.0),  # Some other event during input window
        (5, 40.0),  # Some other event during gap window
        (4, 72.0),  # Death after gap window
    ]


@pytest.fixture
def successful_discharge_sequence():
    """Patient admitted to ICU, survives gap period, then discharged."""
    return [
        (1, 0.0),  # ICU admission at t=0
        (5, 20.0),  # Some other event during input window
        (5, 40.0),  # Some other event during gap window
        (3, 72.0),  # Hospital discharge after gap window
    ]


@pytest.fixture
def impossible_readmission_sequence():
    """Patient readmitted to ICU during gap period."""
    return [
        (1, 0.0),  # Initial ICU admission
        (5, 12.0),  # Other event
        (1, 24.0),  # Another ICU admission during gap
        (4, 72.0),  # Death (but sequence already failed)
    ]


@pytest.fixture
def undetermined_sequence():
    """Patient with no conclusive outcome."""
    return [
        (1, 0.0),  # ICU admission
        (5, 24.0),  # Other event at input window boundary
        (5, 48.0),  # Other event at gap window boundary
        (5, 72.0),  # Other event (no death/discharge)
    ]


def test_basic_task_outcomes(
    task_config_yaml,
    token_map,
    successful_death_sequence,
    successful_discharge_sequence,
    impossible_readmission_sequence,
    undetermined_sequence,
):
    """Test basic task outcomes with different sequence patterns."""
    # Create config from YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(task_config_yaml)
        f.flush()
        task_config = create_zero_shot_task(f.name, token_map)

    # Create model with test sequences
    test_sequences = [
        successful_death_sequence,
        successful_discharge_sequence,
        impossible_readmission_sequence,
        undetermined_sequence,
    ]
    model = DummyModel(test_sequences)

    # Create dummy prompts
    batch_size = len(test_sequences)
    prompts = torch.zeros((batch_size, 1), dtype=torch.long)

    # Generate sequences
    sequences, satisfied, impossible = generate(model=model, prompts=prompts, task_config=task_config)

    # Check outcomes
    assert satisfied[0]  # Death sequence succeeded
    assert satisfied[1]  # Discharge sequence succeeded
    assert impossible[2]  # Readmission sequence failed
    assert not satisfied[3] and not impossible[3]  # Undetermined sequence

    # Check sequence lengths
    assert len(sequences[0]) == len(successful_death_sequence)
    assert len(sequences[1]) == len(successful_discharge_sequence)
    assert len(sequences[2]) <= len(impossible_readmission_sequence)  # Should stop early
    assert len(sequences[3]) == len(undetermined_sequence)

    # Check token patterns
    assert sequences[0, -1] == 4  # Ends with death
    assert sequences[1, -1] == 3  # Ends with discharge
    assert sequences[2, 2] == 1  # Failed on readmission


def test_time_edge_cases():
    """Test sequences with events exactly at window boundaries."""
    # TODO: Add tests for:
    # - Events exactly at 24h boundary
    # - Events exactly at 48h boundary
    # - Events just inside/outside boundaries


def test_budget_interactions():
    """Test interaction between budget and task constraints."""
    # TODO: Add tests for:
    # - Max sequence length reached before conclusion
    # - Max time reached before conclusion
    # - Budget hit after task already determined


def test_batch_processing():
    """Test batch processing with mixed sequence statuses."""
    # TODO: Add tests for:
    # - Sequences finishing at different times
    # - Some sequences hitting budget while others complete normally
    # - All sequences completing simultaneously
