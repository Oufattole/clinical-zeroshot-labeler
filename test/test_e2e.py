import tempfile

import polars as pl
import pytest
import torch
from loguru import logger

from czsl.config import TaskExtractorConfig
from czsl.labeler import (
    AutoregressiveWindowTree,
    EventBound,
    TemporalBound,
    WindowNode,
    calculate_index_timestamp_info,
    convert_task_config,
    timedelta,
)


def print_window_tree(node, indent="", is_last=True):
    """Print a WindowNode and its subtree in a hierarchical tree format.

    Args:
        node (WindowNode): The node to print
        indent (str): The current indentation string
        is_last (bool): Whether this node is the last child of its parent
    """
    # Print current node with proper indentation
    branch = "└── " if is_last else "├── "
    logger.info(f"{indent}{branch}{node.name}")

    # Prepare indentation for children
    child_indent = indent + ("    " if is_last else "│   ")

    # Print all children
    for i, child in enumerate(node.children):
        is_last_child = i == len(node.children) - 1
        print_window_tree(child, child_indent, is_last_child)


def print_window_tree_with_state(node, batch_idx=0, indent="", is_last=True, time=None):
    """Print a WindowNode and its subtree including state information.

    Args:
        node (WindowNode): The node to print
        batch_idx (int): The batch index to show state for
        indent (str): The current indentation string
        is_last (bool): Whether this node is the last child of its parent
    """
    # Get state information if available
    state_info = ""
    if hasattr(node, "batch_states") and len(node.batch_states) > batch_idx:
        state = node.batch_states[batch_idx]
        state_info = (
            f" [status={state.status}, "
            f"start={state.start_time}, "
            f"end={state.end_time}, "
            f"counts={state.predicate_counts}] "
            f"label={node.label} ",
            f"index_timestamp={node.index_timestamp}",
        )

    # Print current node with proper indentation and state
    branch = "└── " if is_last else "├── "
    logger.info(f"{indent}{branch}{node.name}{state_info}")

    # Prepare indentation for children
    child_indent = indent + ("    " if is_last else "│   ")

    # Print all children
    for i, child in enumerate(node.children):
        is_last_child = i == len(node.children) - 1
        print_window_tree_with_state(child, batch_idx, child_indent, is_last_child)


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
        times = []
        for i, seq in enumerate(self.sequences):
            if self.current_positions[i] < len(seq):
                tokens.append(seq[self.current_positions[i]][0])
                times.append(seq[self.current_positions[i]][1])
            else:
                raise ValueError("Sequence is exhausted, zero_shot_labeler should have stopped generation.")
            self.current_positions[i] += 1
        next_codes = torch.tensor(tokens, device=prompts.device)
        next_times = torch.tensor(times, device=prompts.device) / 24
        next_numeric_values = torch.zeros_like(next_codes)
        logger.warning(f"times: {next_times}")
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
def metadata_df():
    return pl.DataFrame(
        {
            "code": [
                "PAD",
                "ICU_ADMISSION//MEDICAL",
                "ICU_DISCHARGE//MEDICAL",
                "HOSPITAL_DISCHARGE//MEDICAL",
                "MEDS_DEATH",
                "OTHER_EVENT",
            ]
        }
    ).with_row_index("code/vocab_index")


@pytest.fixture
def successful_death_sequence():
    """Patient admitted to ICU, survives gap period, then dies."""
    return [
        (5, 0.0),  # Some other event at index timestamp
        (5, 20.0),  # Some other event during input window
        (5, 40.0),  # Some other event during gap window
        (4, 72.0),  # Death after gap window
        (5, 73.0),  # Some other event after death
    ]


@pytest.fixture
def successful_discharge_sequence():
    """Patient admitted to ICU, survives gap period, then discharged."""
    return [
        (5, 0.0),  # Some other event at index timestamp
        (5, 20.0),  # Some other event during input window
        (5, 40.0),  # Some other event during gap window
        (3, 72.0),  # Hospital discharge after gap window
        (5, 73.0),  # Some other event after death
    ]


@pytest.fixture
def impossible_readmission_sequence():
    """Patient readmitted to ICU during gap period."""
    return [
        (5, 0.0),  # Some other event at index timestamp
        (5, 12.0),  # Other event
        (1, 24.0),  # Another ICU admission during gap
        (4, 72.0),  # Death (but sequence already failed)
        (5, 73.0),  # Some other event after death
    ]


@pytest.fixture
def undetermined_sequence():
    """Patient with no conclusive outcome."""
    return [
        (5, 0.0),  # Some other event at index timestamp
        (5, 24.0),  # Other event at input window boundary
        (5, 48.0),  # Other event at gap window boundary
        (5, 72.0),  # Other event (no death/discharge)
        (5, 73.0),  # Some other event after death
    ]


def test_window_tree():
    """Test the window tree implementation."""
    # Create tree for in-hospital mortality prediction
    tensorized_predicates = {
        "Admission": torch.tensor([1]),
        "Lab": torch.tensor([2]),
        "Death": torch.tensor([3]),
    }
    root = WindowNode(
        name="trigger",
        start_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        end_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        predicate_constraints={"Admission": (1, 1)},  # Admission token = 1
        index_timestamp=None,
        label=None,
        tensorized_predicates=tensorized_predicates,
    )

    obs_window = WindowNode(
        name="observation",
        start_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        end_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(hours=24)),
        predicate_constraints={"Lab": (1, None)},  # Lab test token = 2
        parent=root,
        index_timestamp=None,
        label=None,
        tensorized_predicates=tensorized_predicates,
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
        index_timestamp=None,
        label=None,
        tensorized_predicates=tensorized_predicates,
    )
    obs_window.children.append(outcome_window)

    # Create tracker with batch size 2
    tracker = AutoregressiveWindowTree(root, batch_size=2)
    print_window_tree_with_state(tracker.root)

    logger.info("\n=== Test Step 1: Admission events ===")
    status = tracker.update(tokens=torch.tensor([1, 1]), time_deltas=torch.tensor([0.0, 0.0]))
    logger.info(f"Expecting: Both sequences start, trigger satisfied. Status is: {status}")
    print_window_tree_with_state(tracker.root)
    assert (status == torch.ones_like(status)).all()

    logger.info("\n=== Test Step 2: Lab test vs other event ===")
    status = tracker.update(
        tokens=torch.tensor([2, 4]),  # Lab test for seq 1, other event for seq 2
        time_deltas=torch.tensor([0.2, 0.2]),
    )
    logger.info(f"Expecting: Seq 1 progresses, Seq 2 stalls. Status is: {status}")
    assert (status == torch.ones_like(status)).all(), status

    logger.info("\n=== Test Step 3: Death vs other event ===")
    status = tracker.update(tokens=torch.tensor([3, 4]), time_deltas=torch.tensor([1.5, 1.5]))
    logger.info(f"Expecting: Seq 1 completes successfully, Seq 2 fails. Status is: {status}")
    assert (status == torch.tensor([2, 3])).all(), status


def test_icu_mortality_successful_death_sequence(
    task_config_yaml,
    metadata_df,
    successful_death_sequence,
):
    """Test basic task outcomes with different sequence patterns."""
    # Create config from YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(task_config_yaml)
        f.flush()
        task_config = TaskExtractorConfig.load(f.name)

    # Create model with test sequences
    test_sequences = [
        successful_death_sequence,
        # successful_discharge_sequence,
        # impossible_readmission_sequence,
        # undetermined_sequence,
    ]
    model = DummyModel(test_sequences)

    # Create dummy prompts and tracker
    batch_size = len(test_sequences)
    prompts = torch.zeros((batch_size, 1), dtype=torch.long)
    tree = convert_task_config(task_config, batch_size=batch_size, metadata_df=metadata_df)
    gap_days, prior_windows, index_window = calculate_index_timestamp_info(tree)
    tree.root.ignore_windows(prior_windows + ["trigger"])

    print_window_tree_with_state(tree.root)

    # Test each step of sequence generation
    def check_step(expected_status):
        next_tokens, next_times, _ = model.generate_next_token(prompts)
        logger.info(f"Tokens: {next_tokens}, Times: {next_times}")
        status = tree.update(tokens=next_tokens, time_deltas=next_times + gap_days)
        print_window_tree_with_state(tree.root)
        logger.info(f"Status: {status}")
        assert torch.equal(status, expected_status)
        return torch.cat([prompts, next_tokens.unsqueeze(1)], dim=1)

    batch_size = 1
    # Step 1: Initial ICU admissions
    # All sequences should start and satisfy trigger
    prompts = check_step(torch.zeros(batch_size))

    # Step 2: Events during input window
    # All sequences should remain active
    prompts = check_step(torch.ones(batch_size))

    # Step 3: Events during/end of gap window
    # Readmission sequence should become impossible
    prompts = check_step(torch.ones(batch_size))

    # Step 4: Final outcomes
    # Death sequence -> satisfied
    # Discharge sequence -> satisfied
    # Readmission sequence -> already impossible
    # Undetermined sequence -> still active/undetermined
    expected = torch.tensor([2])  # Satisfied, Satisfied, Impossible, Active
    prompts = check_step(expected)

    expected = torch.tensor([2])  # Satisfied, Satisfied, Impossible, Active
    prompts = check_step(expected)

    logger.info("Test completed - all sequence patterns behaved as expected")


def test_icu_mortality_successful_discharge_sequence(
    task_config_yaml,
    metadata_df,
    successful_discharge_sequence,
):
    """Test basic task outcomes with different sequence patterns."""
    # Create config from YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(task_config_yaml)
        f.flush()
        task_config = TaskExtractorConfig.load(f.name)

    # Create model with test sequences
    test_sequences = [
        successful_discharge_sequence,
    ]
    model = DummyModel(test_sequences)

    # Create dummy prompts and tracker
    batch_size = len(test_sequences)
    prompts = torch.zeros((batch_size, 1), dtype=torch.long)
    tree = convert_task_config(task_config, batch_size=batch_size, metadata_df=metadata_df)
    gap_days, prior_windows, index_window = calculate_index_timestamp_info(tree)
    tree.root.ignore_windows(prior_windows + ["trigger"])

    print_window_tree_with_state(tree.root)

    # Test each step of sequence generation
    def check_step(expected_status):
        next_tokens, next_times, _ = model.generate_next_token(prompts)
        logger.info(f"Tokens: {next_tokens}, Times: {next_times}")
        status = tree.update(tokens=next_tokens, time_deltas=next_times + gap_days)
        print_window_tree_with_state(tree.root)
        logger.info(f"Status: {status}")
        assert torch.equal(status, expected_status)
        return torch.cat([prompts, next_tokens.unsqueeze(1)], dim=1)

    batch_size = 1
    # Step 1: Initial ICU admissions
    # All sequences should start and satisfy trigger
    prompts = check_step(torch.zeros(batch_size))

    # Step 2: Events during input window
    # All sequences should remain active
    prompts = check_step(torch.ones(batch_size))

    # Step 3: Events during/end of gap window
    # Readmission sequence should become impossible
    prompts = check_step(torch.ones(batch_size))

    # Step 4: Final outcomes
    # Discharge sequence -> satisfied
    expected = torch.tensor([2])  # Satisfied
    prompts = check_step(expected)

    expected = torch.tensor([2])  # Satisfied
    prompts = check_step(expected)

    logger.info("Test completed - all sequence patterns behaved as expected")


def test_icu_mortality_impossible_readmission_sequence(
    task_config_yaml,
    metadata_df,
    impossible_readmission_sequence,
):
    """Test basic task outcomes with different sequence patterns."""
    # Create config from YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(task_config_yaml)
        f.flush()
        task_config = TaskExtractorConfig.load(f.name)

    # Create model with test sequences
    test_sequences = [
        impossible_readmission_sequence,
    ]
    model = DummyModel(test_sequences)

    # Create dummy prompts and tracker
    batch_size = len(test_sequences)
    prompts = torch.zeros((batch_size, 1), dtype=torch.long)
    tree = convert_task_config(task_config, batch_size=batch_size, metadata_df=metadata_df)
    gap_days, prior_windows, index_window = calculate_index_timestamp_info(tree)
    tree.root.ignore_windows(prior_windows + ["trigger"])

    print_window_tree_with_state(tree.root)

    # Test each step of sequence generation
    def check_step(expected_status):
        next_tokens, next_times, _ = model.generate_next_token(prompts)
        logger.info(f"Tokens: {next_tokens}, Times: {next_times}")
        status = tree.update(tokens=next_tokens, time_deltas=next_times + gap_days)
        print_window_tree_with_state(tree.root)
        logger.info(f"Status: {status}")
        assert torch.equal(status, expected_status)
        return torch.cat([prompts, next_tokens.unsqueeze(1)], dim=1)

    batch_size = 1
    # Step 1: Initial ICU admissions
    # All sequences should start and satisfy trigger
    prompts = check_step(torch.zeros(batch_size))

    # Step 2: Events during input window
    # All sequences should remain active
    prompts = check_step(torch.ones(batch_size))

    # Step 3: Events during/end of gap window
    # Readmission sequence should become impossible
    expected = torch.tensor([3])  # Impossible
    prompts = check_step(expected)

    # Step 4: Final outcomes
    # Death sequence -> satisfied
    # Readmission sequence -> already impossible
    expected = torch.tensor([3])  # Impossible
    prompts = check_step(expected)

    expected = torch.tensor([3])  # Impossible
    prompts = check_step(expected)

    logger.info("Test completed - all sequence patterns behaved as expected")


def test_icu_mortality_undetermined_sequence(
    task_config_yaml,
    metadata_df,
    undetermined_sequence,
):
    """Test basic task outcomes with different sequence patterns."""
    # Create config from YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(task_config_yaml)
        f.flush()
        task_config = TaskExtractorConfig.load(f.name)

    # Create model with test sequences
    test_sequences = [
        undetermined_sequence,
    ]
    model = DummyModel(test_sequences)

    # Create dummy prompts and tracker
    batch_size = len(test_sequences)
    prompts = torch.zeros((batch_size, 1), dtype=torch.long)
    tree = convert_task_config(task_config, batch_size=batch_size, metadata_df=metadata_df)
    gap_days, prior_windows, index_window = calculate_index_timestamp_info(tree)
    tree.root.ignore_windows(prior_windows + ["trigger"])

    print_window_tree_with_state(tree.root)

    # Test each step of sequence generation
    def check_step(expected_status):
        next_tokens, next_times, _ = model.generate_next_token(prompts)
        logger.info(f"Tokens: {next_tokens}, Times: {next_times}")
        status = tree.update(tokens=next_tokens, time_deltas=next_times + gap_days)
        print_window_tree_with_state(tree.root)
        logger.info(f"Status: {status}")
        assert torch.equal(status, expected_status)
        return torch.cat([prompts, next_tokens.unsqueeze(1)], dim=1)

    batch_size = 1
    # Step 1: Initial ICU admissions
    # All sequences should start and satisfy trigger
    prompts = check_step(torch.zeros(batch_size))

    # Step 2: Events during input window
    # All sequences should remain active
    prompts = check_step(torch.ones(batch_size))

    # Step 3: Events during/end of gap window
    # Readmission sequence should become impossible
    prompts = check_step(torch.ones(batch_size))

    # Step 4: Final outcomes
    # Undetermined sequence -> still active/undetermined
    expected = torch.tensor([1])  # Active
    prompts = check_step(expected)

    # TODO fix this, it should be impossible
    expected = torch.tensor([1])  # Active
    prompts = check_step(expected)

    logger.info("Test completed - all sequence patterns behaved as expected")


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
