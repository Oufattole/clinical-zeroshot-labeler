import tempfile

import polars as pl
import pytest
import torch
from aces.config import TaskExtractorConfig
from loguru import logger

from clinical_zeroshot_labeler.labeler import (
    AutoregressiveWindowTree,
    EventBound,
    PredicateTensor,
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
        numeric_values = []
        for i, seq in enumerate(self.sequences):
            if self.current_positions[i] < len(seq):
                tokens.append(seq[self.current_positions[i]][0])
                times.append(seq[self.current_positions[i]][1])
                numeric_values.append(seq[self.current_positions[i]][2])
            else:
                raise ValueError("Sequence is exhausted, zero_shot_labeler should have stopped generation.")
            self.current_positions[i] += 1
        next_codes = torch.tensor(tokens, device=prompts.device)
        next_times = torch.tensor(times, device=prompts.device) / 24
        next_numeric_values = torch.tensor(numeric_values)
        logger.warning(f"times: {next_times}")
        return next_codes.flatten(), next_times.flatten(), next_numeric_values.flatten()


@pytest.fixture
def icu_morality_task_config_yaml():
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
def abnormal_lab_task_config_yaml():
    return """
    predicates:
        hospital_discharge:
            code: {regex: "HOSPITAL_DISCHARGE//.*"}
        lab:
            code: {regex: "LAB//.*"}
        high_lab:
            code: {regex: "LAB//.*"}
            value_min: 2.0
            value_min_inclusive: True
        low_lab:
            code: {regex: "LAB//.*"}
            value_max: -2.0
            value_max_inclusive: False
        abnormal_lab:
            expr: or(high_lab, low_lab)

    trigger: hospital_discharge

    windows:
        input:
            start: NULL
            end: trigger
            start_inclusive: True
            end_inclusive: True
            index_timestamp: end
        target:
            start: input.end
            end: start + 4d
            start_inclusive: False
            end_inclusive: True
            has:
                lab: (1, None)
            label: abnormal_lab
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
                "LAB//1",
                "LAB//2",
            ]
        }
    ).with_row_index("code/vocab_index")


def test_window_tree():
    """Test the window tree implementation."""
    # Create PredicateTensor objects for each event type
    tensorized_predicates = {
        "Admission": PredicateTensor(
            name="Admission",
            tokens=torch.tensor([1]),
            value_limits=(None, None),
            value_inclusions=(None, None),
            children=[],
            is_and=False,
        ),
        "Lab": PredicateTensor(
            name="Lab",
            tokens=torch.tensor([2]),
            value_limits=(None, None),
            value_inclusions=(None, None),
            children=[],
            is_and=False,
        ),
        "Death": PredicateTensor(
            name="Death",
            tokens=torch.tensor([3]),
            value_limits=(None, None),
            value_inclusions=(None, None),
            children=[],
            is_and=False,
        ),
    }

    # Create root node (trigger window)
    root = WindowNode(
        name="trigger",
        start_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        end_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        predicate_constraints={"Admission": (1, 1)},  # Must see exactly one admission
        index_timestamp=None,
        label=None,
        tensorized_predicates=tensorized_predicates,
    )

    # Create observation window
    obs_window = WindowNode(
        name="observation",
        start_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(0)),
        end_bound=TemporalBound(reference="trigger", inclusive=True, offset=timedelta(hours=24)),
        predicate_constraints={"Lab": (1, None)},  # At least one lab test
        parent=root,
        index_timestamp=None,
        label=None,
        tensorized_predicates=tensorized_predicates,
    )
    root.children.append(obs_window)

    # Create outcome window
    outcome_window = WindowNode(
        name="outcome",
        start_bound=TemporalBound(reference="observation.end", inclusive=True, offset=timedelta(0)),
        end_bound=EventBound(
            reference="observation.end",
            inclusive=True,
            predicate=tensorized_predicates["Death"],  # Use Death predicate object
            direction="next",
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
    status = tracker.update(
        tokens=torch.tensor([1, 1]),
        time_deltas=torch.tensor([0.0, 0.0]),
        numeric_values=torch.tensor([0.0, 0.0]),
    )
    logger.info(f"Expecting: Both sequences start, trigger satisfied. Status is: {status}")
    print_window_tree_with_state(tracker.root)
    assert (status == torch.ones_like(status)).all()

    logger.info("\n=== Test Step 2: Lab test vs other event ===")
    status = tracker.update(
        tokens=torch.tensor([2, 4]),  # Lab test for seq 1, other event for seq 2
        time_deltas=torch.tensor([0.2, 0.2]),
        numeric_values=torch.tensor([0.0, 0.0]),
    )
    logger.info(f"Expecting: Seq 1 progresses, Seq 2 stalls. Status is: {status}")
    assert (status == torch.ones_like(status)).all(), status

    logger.info("\n=== Test Step 3: Death vs other event ===")
    status = tracker.update(
        tokens=torch.tensor([3, 4]),  # Death for seq 1, other event for seq 2
        time_deltas=torch.tensor([1.5, 1.5]),
        numeric_values=torch.tensor([0.0, 0.0]),
    )
    logger.info(f"Expecting: Seq 1 active, Seq 2 fails. Status is: {status}")
    assert (status == torch.tensor([1, 3])).all(), status

    logger.info("\n=== Test Step 4: Death vs other event ===")
    status = tracker.update(
        tokens=torch.tensor([5, 5]),  # Death for seq 1, other event for seq 2
        time_deltas=torch.tensor([2.0, 2.0]),
        numeric_values=torch.tensor([0.0, 0.0]),
    )
    logger.info(f"Expecting: Seq 1 completes successfully, Seq 2 fails. Status is: {status}")
    assert (status == torch.tensor([2, 3])).all(), status


@pytest.fixture
def successful_death_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 20.0, 0.0),  # Other event during input
            (5, 40.0, 0.0),  # Other event during gap
            (4, 72.0, 0.0),  # Death after gap
            (5, 73.0, 0.0),  # Other event after death
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Gap window active
            torch.tensor([1]),  # Satisfied (death)
            torch.tensor([2]),  # Remains satisfied
        ],
        "label": True,
    }


@pytest.fixture
def successful_discharge_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 20.0, 0.0),  # Other event during input
            (5, 40.0, 0.0),  # Other event during gap
            (3, 72.0, 0.0),  # Hospital discharge after gap
            (5, 73.0, 0.0),  # Other event after discharge
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Gap window active
            torch.tensor([1]),  # Satisfied (discharge)
            torch.tensor([2]),  # Remains satisfied
        ],
        "label": False,
    }


@pytest.fixture
def impossible_readmission_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 12.0, 0.0),  # Other event
            (1, 24.0, 0.0),  # ICU readmission during gap
            (4, 72.0, 0.0),  # Death (but already failed)
            (5, 73.0, 0.0),  # Other event after death
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([3]),  # Impossible (readmission)
            torch.tensor([3]),  # Remains impossible
            torch.tensor([3]),  # Remains impossible
        ],
        "label": False,
    }


@pytest.fixture
def undetermined_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 24.0, 0.0),  # Other event at input boundary
            (5, 48.0, 0.0),  # Other event at gap boundary
            (5, 72.0, 0.0),  # Other event (no outcome)
            (5, 73.0, 0.0),  # Other event (no outcome)
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Gap window active
            torch.tensor([1]),  # Still active (no outcome)
            torch.tensor([1]),  # Still active (no outcome)
        ],
        "label": False,
    }


@pytest.fixture
def exact_boundary_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 24.0, 0.0),  # Event exactly at input window boundary
            (5, 48.0, 0.0),  # Event exactly at gap window boundary
            (4, 72.1, 0.0),  # Death just after minimum time
            (5, 72.1, 0.0),  # Death just after minimum time
            (5, 72.2, 0.0),  # Random event after
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Gap window active
            torch.tensor([1]),  # Waiting for next time token
            torch.tensor([1]),  # Waiting for next time token
            torch.tensor([2]),  # Satisfied
        ],
        "label": True,
    }


@pytest.fixture
def boundary_exclusion_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 24.0, 0.0),  # Event at input boundary (excluded)
            (4, 48.0, 0.0),  # Death at gap boundary (excluded)
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([3]),  # Impossible (death during gap)
        ],
        "label": False,
    }


@pytest.fixture
def death_after_discharge_same_time_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 24.0, 0.0),  # Event at input boundary (excluded)
            (5, 73.0, 0.0),  # Random Event
            (4, 74.0, 0.0),  # Discharge
            (3, 74.0, 0.0),  # Death
            (5, 75.0, 0.0),  # Event at later time
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Input window active
            torch.tensor([2]),  # Input window active
        ],
        "label": True,
    }


@pytest.fixture
def death_before_discharge_same_time_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 24.0, 0.0),  # Event at input boundary (excluded)
            (5, 73.0, 0.0),  # Random Event
            (3, 74.0, 0.0),  # Death
            (4, 74.0, 0.0),  # Discharge
            (5, 75.0, 0.0),  # Event at later time
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Input window active
            torch.tensor([2]),  # Input window active
        ],
        "label": True,
    }


@pytest.fixture
def multiple_sequences():
    return {
        "sequence": [
            ((5, 5), (0.0, 0.0), (0.0, 0.0)),  # Other event at index
            ((5, 5), (20.0, 20.0), (0.0, 0.0)),  # Other event during input
            ((5, 1), (40.0, 40.0), (0.0, 0.0)),  # Other event during gap
            ((3, 4), (72.0, 72.0), (0.0, 0.0)),  # Hospital discharge after gap
            ((5, 5), (73.0, 73.0), (0.0, 0.0)),  # Other event after discharge
        ],
        "expected_statuses": [
            torch.tensor([0, 0]),  # Initial state
            torch.tensor([1, 1]),  # Input window active
            torch.tensor([1, 3]),  # Gap window active
            torch.tensor([1, 3]),  # Satisfied (discharge)
            torch.tensor([2, 3]),  # Remains satisfied
        ],
        "label": [False, False],
    }


@pytest.mark.parametrize(
    "sequence_fixture_name",
    [
        "successful_death_sequence",
        "successful_discharge_sequence",
        "impossible_readmission_sequence",
        "undetermined_sequence",
        "exact_boundary_sequence",
        "boundary_exclusion_sequence",
        "death_before_discharge_same_time_sequence",
        "multiple_sequences",
    ],
)
def test_icu_mortality_sequences(icu_morality_task_config_yaml, metadata_df, sequence_fixture_name, request):
    """Test ICU mortality task with different sequence patterns."""
    # Get sequence data from fixture
    sequence_data = request.getfixturevalue(sequence_fixture_name)
    sequence = sequence_data["sequence"]
    expected_statuses = sequence_data["expected_statuses"]
    batch_size = 1 if isinstance(sequence[0][0], int) else len(sequence[0][0])

    # Create config from YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(icu_morality_task_config_yaml)
        f.flush()
        task_config = TaskExtractorConfig.load(f.name)

    # Setup model and tracker
    model = DummyModel([sequence])
    prompts = torch.zeros((batch_size), dtype=torch.long)
    tree = convert_task_config(task_config, batch_size=batch_size, metadata_df=metadata_df)
    gap_days, prior_windows, _ = calculate_index_timestamp_info(tree)
    tree.root.initialize_batch(batch_size)
    tree.root.ignore_windows(prior_windows + ["trigger"])
    logger.info(print_window_tree_with_state(tree.root))

    # Test each step
    status = None
    for step, expected_status in enumerate(expected_statuses):
        next_tokens, next_times, numeric_values = model.generate_next_token(prompts)
        time_deltas = next_times + gap_days
        status = tree.update(
            tokens=next_tokens.flatten(),
            time_deltas=time_deltas.flatten(),
            numeric_values=numeric_values.flatten(),
        )
        logger.info("Added Event: " + str(next_tokens))
        logger.info("TREE NAME: " + tree.root.children[0].children[0].children[0].name)
        logger.info("Time: " + str(time_deltas))
        print_window_tree_with_state(tree.root)

        assert torch.equal(
            status, expected_status
        ), f"{sequence_fixture_name} - Step {step}: Expected status {expected_status}, got {status}"

        # import pdb; pdb.set_trace()
        prompts = torch.vstack([prompts, next_tokens])
    assert (
        torch.tensor(sequence_data["label"]) == (tree.root.get_labels() & (status == torch.tensor([2])))
    ).all()


@pytest.fixture
def successful_abnormal_lab():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 20.0, 0.0),  # Other event during input
            (5, 40.0, 0.0),  # Other event during gap
            (6, 72.0, 100.0),  # Lab event
            (7, 73.0, 100.0),  # Other lab event
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Other event
            torch.tensor([2]),  # Satisfied
            torch.tensor([2]),  # Remains satisfied
        ],
        "label": True,
    }


@pytest.fixture
def successful_second_abnormal_lab():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 20.0, 0.0),  # Other event during input
            (5, 40.0, 0.0),  # Other event during gap
            (6, 72.0, 0.0),  # Normal Lab event
            (7, 73.0, 100.0),  # Abnormal lab event
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Other event
            torch.tensor([1]),  # Active
            torch.tensor([2]),  # Remains satisfied
        ],
        "label": True,
    }


@pytest.fixture
def normal_lab_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 20.0, 0.0),  # Other event during input
            (6, 72.0, 1.5),  # Lab event with normal value (between -2 and 2)
            (7, 73.0, -1.8),  # Another normal lab
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Still active (no abnormal labs)
            torch.tensor([1]),  # Still active (no abnormal labs)
        ],
        "label": False,  # No abnormal labs found
    }


@pytest.fixture
def edge_case_lab_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (7, 73.0, -2.0),  # Lab exactly at low threshold (exclusive)
            (6, 72.0, 2.0),  # Lab exactly at high threshold (inclusive)
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([1]),  # Still active
            torch.tensor([2]),  # Satisfied (hit inclusive high threshold)
        ],
        "label": True,  # Abnormal lab found (high threshold inclusive)
    }


@pytest.mark.parametrize(
    "sequence_fixture_name",
    [
        "successful_abnormal_lab",
        "successful_second_abnormal_lab",
        "normal_lab_sequence",
        "edge_case_lab_sequence",
    ],
)
def test_abnormal_lab_sequences(abnormal_lab_task_config_yaml, metadata_df, sequence_fixture_name, request):
    """Test ICU mortality task with different sequence patterns."""
    # Get sequence data from fixture
    sequence_data = request.getfixturevalue(sequence_fixture_name)
    sequence = sequence_data["sequence"]
    expected_statuses = sequence_data["expected_statuses"]

    # Create config from YAML
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write(abnormal_lab_task_config_yaml)
        f.flush()
        task_config = TaskExtractorConfig.load(f.name)

    # Setup model and tracker
    model = DummyModel([sequence])
    batch_size = 1
    prompts = torch.zeros((batch_size, 1), dtype=torch.long)
    tree = convert_task_config(task_config, batch_size=batch_size, metadata_df=metadata_df)
    gap_days, prior_windows, index_window = calculate_index_timestamp_info(tree)
    tree.root.ignore_windows(prior_windows + ["trigger"])

    # Test each step
    for step, expected_status in enumerate(expected_statuses):
        logger.info(f"Step {step}: Expected status {expected_status}")
        next_tokens, next_times, numeric_values = model.generate_next_token(prompts)
        status = tree.update(
            tokens=next_tokens, time_deltas=next_times + gap_days, numeric_values=numeric_values
        )

        assert torch.equal(
            status, expected_status
        ), f"{sequence_fixture_name} - Step {step}: Expected status {expected_status}, got {status}"

        prompts = torch.cat([prompts, next_tokens.unsqueeze(1)], dim=1)
    assert sequence_data["label"] == any(tree.root.get_labels())
