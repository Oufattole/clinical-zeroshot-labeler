import pytest
import torch
from loguru import logger

from czsl.labeler import (
    AutoregressiveWindowTree,
    EventBound,
    TemporalBound,
    WindowNode,
    timedelta,
)


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
                raise ValueError("Sequence is exhausted, zero_shot_labeler should have stopped generation.")
        next_codes = torch.tensor(tokens, device=prompts.device)
        next_times = torch.ones_like(next_codes) / 365
        next_numeric_values = torch.zeros_like(next_codes)
        return next_codes, next_times, next_numeric_values


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
        start: input.end
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


def test_window_tree():
    """Test the window tree implementation."""
    # Create tree for in-hospital mortality prediction
    root = WindowNode(
        name="trigger",
        start_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        end_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        predicate_constraints={"1": (1, 1)},  # Admission token = 1
    )

    obs_window = WindowNode(
        name="observation",
        start_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        end_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(hours=24)),
        predicate_constraints={"2": (1, None)},  # Lab test token = 2
        parent=root,
    )
    root.children.append(obs_window)

    outcome_window = WindowNode(
        name="outcome",
        start_bound=TemporalBound(reference="observation.end", inclusive=True, offset=timedelta(0)),
        end_bound=EventBound(
            reference="observation.end", inclusive=True, predicate="3", direction="next"  # Death token
        ),
        predicate_constraints={},
        parent=obs_window,
    )
    obs_window.children.append(outcome_window)

    # Create tracker with batch size 2
    tracker = AutoregressiveWindowTree(root, batch_size=2)

    logger.info("\n=== Test Step 1: Admission events ===")
    status = tracker.update(tokens=torch.tensor([1, 1]), time_deltas=torch.tensor([0.0, 0.0]))
    logger.info(f"Expecting: Both sequences start, trigger satisfied. Status is: {status}")
    assert (status == torch.ones_like(status)).all()

    logger.info("\n=== Test Step 2: Lab test vs other event ===")
    status = tracker.update(
        tokens=torch.tensor([2, 4]),  # Lab test for seq 1, other event for seq 2
        time_deltas=torch.tensor([0.2, 0.2]),
    )
    logger.info(f"Expecting: Seq 1 progresses, Seq 2 stalls. Status is: {status}")
    assert (status == torch.ones_like(status)).all(), status

    print("\n=== Test Step 3: Death vs other event ===")
    status = tracker.update(tokens=torch.tensor([3, 4]), time_deltas=torch.tensor([1.5, 1.5]))
    logger.info(f"Expecting: Seq 1 completes successfully, Seq 2 fails. Status is: {status}")
    assert (status == torch.tensor([2, 3])).all(), status


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
    # with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
    #     f.write(task_config_yaml)
    #     f.flush()
    #     task_config = TaskExtractorConfig.load(f.name)
    # import pdb; pdb.set_trace()

    # zs_task_config = ZeroShotTaskConfig(task_config, token_map, max_seq_len=10)

    # # Create model with test sequences
    # test_sequences = [
    #     successful_death_sequence,
    #     successful_discharge_sequence,
    #     impossible_readmission_sequence,
    #     undetermined_sequence,
    # ]
    # model = DummyModel(test_sequences)

    # # Create dummy prompts
    # batch_size = len(test_sequences)
    # prompts = torch.zeros((batch_size, 1), dtype=torch.long)

    # # Generate sequences
    # generation_output = generate(
    #     model=model,
    #     prompts=prompts,
    #     zs_task_config=zs_task_config,
    #     end_time_delta=torch.zeros((batch_size), dtype=torch.float32),
    # )
    # sequences = generation_output.sequences
    # satisfied = generation_output.satisfied
    # impossible = generation_output.impossible
    # _ = generation_output.times

    # # Check outcomes
    # assert satisfied[0]  # Death sequence succeeded
    # assert satisfied[1]  # Discharge sequence succeeded
    # assert impossible[2]  # Readmission sequence failed
    # assert not satisfied[3] and not impossible[3]  # Undetermined sequence

    # # Check sequence lengths
    # assert len(sequences[0]) == len(successful_death_sequence)
    # assert len(sequences[1]) == len(successful_discharge_sequence)
    # assert len(sequences[2]) <= len(impossible_readmission_sequence)  # Should stop early
    # assert len(sequences[3]) == len(undetermined_sequence)

    # # Check token patterns
    # assert sequences[0, -1] == 4  # Ends with death
    # assert sequences[1, -1] == 3  # Ends with discharge
    # assert sequences[2, 2] == 1  # Failed on readmission


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
