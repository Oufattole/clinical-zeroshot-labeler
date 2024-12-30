import polars as pl
import pytest
import torch
from loguru import logger

from clinical_zeroshot_labeler.labeler import (
    AutoregressiveWindowTree,
    EventBound,
    PredicateTensor,
    SequenceLabeler,
    TemporalBound,
    WindowNode,
    timedelta,
)

SUCCESSFUL_DISCHARGE_SEQUENCE = {
    "sequence": [
        (5, 0.0, 0.0),  # Other event at index
        (5, 20.0, 0.0),  # Other event during input
        (5, 40.0, 0.0),  # Other event during gap
        (5, 72.0, 0.0),  # Random Event at gap end
        (3, 73.0, 0.0),  # Hospital discharge after gap
        (5, 74.0, 0.0),  # Other event after discharge
    ],
    "expected_statuses": [
        torch.tensor([0]),  # Initial state
        torch.tensor([1]),  # Input window active
        torch.tensor([1]),  # Gap window active
        torch.tensor([1]),  # Random Event after gap
        torch.tensor([1]),  # Satisfied (discharge)
        torch.tensor([2]),  # Remains satisfied
    ],
    "label": False,
}

IMPOSSIBLE_READMISSION_SEQUENCE = {
    "sequence": [
        (5, 0.0, 0.0),  # Other event at index
        (5, 12.0, 0.0),  # Other event
        (1, 25.0, 0.0),  # ICU readmission during gap
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
def multiple_sequences_death():
    seq1, seq2 = SUCCESSFUL_DISCHARGE_SEQUENCE, IMPOSSIBLE_READMISSION_SEQUENCE
    # Get the longest status list
    expected_statuses = [seq1["expected_statuses"], seq2["expected_statuses"]]
    max_len = max(len(statuses) for statuses in expected_statuses)

    # Convert to list of tensors, one for each step
    status_tensors = []
    for step in range(max_len):
        # For each step, get status from each example
        # If past the end, use the final status
        statuses = [
            status_list[step].item() if step < len(status_list) else status_list[-1].item()
            for status_list in expected_statuses
        ]
        status_tensors.append(torch.tensor(statuses))

    return {
        "sequence": [seq1["sequence"], seq2["sequence"]],
        "expected_statuses": status_tensors,
        "label": [seq1["label"], seq2["label"]],
    }


def convert_sequence_times(sequence, time_scale):
    """Convert sequence times to the specified scale."""
    if time_scale == "Y":
        conversion_factor = 1 / 24 / 365
    elif time_scale == "D":
        conversion_factor = 1 / 24
    else:
        raise NotImplementedError(f"Invalid time scale: {time_scale}")

    if isinstance(sequence[0], tuple):
        return [[(t, time * conversion_factor, v) for t, time, v in sequence]]
    else:  # Multiple sequences
        output_sequence = []
        for s in sequence:
            output_sequence.append([(t, time * conversion_factor, v) for t, time, v in s])
        return output_sequence


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
    if not node.ignore:
        state = node.state
        state_info = (
            f" [status={state.status.tolist()}, "
            f"start={state.start_time.tolist()}, "
            f"end={state.end_time.tolist()}, "
            f"counts={state.predicate_counts}] "
            f"label={node.label} ",
            f"index_timestamp={node.index_timestamp}",
        )

    # Print current node with proper indentation and state
    branch = "└── " if is_last else "├── "
    print(f"{indent}{branch}{node.name}{state_info}")

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
def alternative_icu_morality_task_config_yaml():
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
        start: trigger  # Direct reference to trigger instead of input.end
        end: start + 72h
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


class SimpleGenerativeModel:
    """A simple mock generative model that returns predefined sequences."""

    def __init__(self, sequences: list[list[tuple[int, float, float]]]):
        """
        Args:
            sequences: List of sequences, where each sequence is a list of
                      (token, time, value) tuples
        """
        self.sequences = sequences
        self.current_positions = [0] * len(sequences)
        self.batch_size = len(sequences)

    def generate_next_token(self, prompt: torch.Tensor):
        """Simulate generating the next token for each sequence in the batch."""
        tokens = []
        times = []
        values = []

        for i, seq in enumerate(self.sequences):
            if self.current_positions[i] < len(seq):
                token, time, value = seq[self.current_positions[i]]
                self.current_positions[i] += 1
            else:
                # For finished sequences, repeat last values
                token, time, value = seq[-1]
            tokens.append(token)
            times.append(time)
            values.append(value)

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(times, dtype=torch.float),
            torch.tensor(values, dtype=torch.float),
        )

    def is_finished(self):
        """Check if all sequences have been fully processed."""
        return all(pos >= len(seq) for pos, seq in zip(self.current_positions, self.sequences))


@pytest.fixture
def impossible_death_boundary_sequence():
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
def successful_death_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 20.0, 0.0),  # Other event during input
            (5, 40.0, 0.0),  # Other event during gap
            (4, 73.0, 0.0),  # Death after gap
            (5, 74.0, 0.0),  # Other event after death
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
            (5, 72.0, 0.0),  # Random Event at gap end
            (3, 73.0, 0.0),  # Hospital discharge after gap
            (5, 74.0, 0.0),  # Other event after discharge
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Gap window active
            torch.tensor([1]),  # Random Event after gap
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
            (1, 25.0, 0.0),  # ICU readmission during gap
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
            (5, 72.0, 0.0),  # Event exactly at gap window boundary
            (4, 72.1, 0.0),  # Death just after minimum time
            (5, 72.1, 0.0),  # Death just after minimum time
            (5, 72.2, 0.0),  # Random event after
        ],
        "expected_statuses": [
            torch.tensor([0]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Gap window active
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


@pytest.fixture
def alt_successful_death_sequence():
    return {
        "sequence": [
            (5, 0.0, 0.0),  # Other event at index
            (5, 20.0, 0.0),  # Other event during input
            (5, 40.0, 0.0),  # Other event during gap
            (5, 73.0, 0.0),  # Other event after gap
            (4, 73.0, 0.0),  # Death after gap
            (5, 74.0, 0.0),  # Other event after death
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([1]),  # Input window active
            torch.tensor([1]),  # Gap window active
            torch.tensor([1]),  # Gap window active
            torch.tensor([1]),  # Satisfied (death)
            torch.tensor([2]),  # Remains satisfied
        ],
        "label": True,
    }


@pytest.fixture
def hematocrit_task_config_yaml():
    return """
    predicates:
      trigger_event:
        code: { regex: "^HOSPITAL_DISCHARGE//.*" }
      lab:
        code: { regex: "^LAB//.*" }
      lab_1:
        code: "LAB//1"
        value_max: 24
        value_max_inclusive: True
      lab_2:
        code: "LAB//2"
        value_max: 24
        value_max_inclusive: True
      abnormal_lab:
        expr: or(lab_1,lab_2)

    trigger: trigger_event

    windows:
      input:
        start: NULL
        end: trigger
        start_inclusive: True
        end_inclusive: True
        index_timestamp: end
      target:
        start: input.end
        end: start + 30d
        start_inclusive: False
        end_inclusive: True
        has:
          lab: (1, None)
        label: abnormal_lab
    """


@pytest.fixture
def hematocrit_normal_sequence():
    return {
        "sequence": [
            (3, 0.0, 0.0),  # Hospital discharge at index
            (6, 15.0, 25.0),  # Normal lab (value > 24)
            (7, 20.0, 30.0),  # Another normal lab
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([1]),  # Still active (no abnormal labs)
            torch.tensor([1]),  # Still active (no abnormal labs)
        ],
        "label": False,  # No abnormal labs found
    }


@pytest.fixture
def hematocrit_abnormal_sequence():
    return {
        "sequence": [
            (3, 0.0, 0.0),  # Hospital discharge at index
            (6, 15.0, 23.0),  # Abnormal lab (value <= 24)
            (7, 20.0, 25.0),  # Normal lab after
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([2]),  # Satisfied (abnormal lab found)
            torch.tensor([2]),  # Remains satisfied
        ],
        "label": True,  # Abnormal lab found
    }


@pytest.fixture
def hematocrit_boundary_sequence():
    return {
        "sequence": [
            (3, 0.0, 0.0),  # Hospital discharge at index
            (6, 30.0, 23.0),  # Lab at exactly 30 days (inclusive)
            (7, 31.0, 22.0),  # Lab after window (should be ignored)
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([2]),  # Satisfied (abnormal lab within window)
            torch.tensor([2]),  # Remains satisfied
        ],
        "label": True,  # Abnormal lab found within window
    }


@pytest.fixture
def hematocrit_threshold_sequence():
    return {
        "sequence": [
            (3, 0.0, 0.0),  # Hospital discharge at index
            (6, 15.0, 24.0),  # Lab exactly at threshold (inclusive)
            (7, 20.0, 24.1),  # Lab just above threshold
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([2]),  # Satisfied (at threshold, inclusive)
            torch.tensor([2]),  # Remains satisfied
        ],
        "label": True,  # Abnormal lab at threshold
    }


@pytest.mark.parametrize(
    ["sequence_fixture_name", "task_yaml_name"],
    [
        ("successful_death_sequence", "icu_morality_task_config_yaml"),
        ("successful_discharge_sequence", "icu_morality_task_config_yaml"),
        ("impossible_readmission_sequence", "icu_morality_task_config_yaml"),
        ("undetermined_sequence", "icu_morality_task_config_yaml"),
        ("exact_boundary_sequence", "icu_morality_task_config_yaml"),
        ("boundary_exclusion_sequence", "icu_morality_task_config_yaml"),
        ("death_after_discharge_same_time_sequence", "icu_morality_task_config_yaml"),
        ("death_before_discharge_same_time_sequence", "icu_morality_task_config_yaml"),
        ("multiple_sequences_death", "icu_morality_task_config_yaml"),
        ("successful_abnormal_lab", "abnormal_lab_task_config_yaml"),
        ("successful_second_abnormal_lab", "abnormal_lab_task_config_yaml"),
        ("normal_lab_sequence", "abnormal_lab_task_config_yaml"),
        ("edge_case_lab_sequence", "abnormal_lab_task_config_yaml"),
        ("alt_successful_death_sequence", "alternative_icu_morality_task_config_yaml"),
        ("successful_discharge_sequence", "alternative_icu_morality_task_config_yaml"),
        ("impossible_readmission_sequence", "alternative_icu_morality_task_config_yaml"),
        ("undetermined_sequence", "alternative_icu_morality_task_config_yaml"),
        ("exact_boundary_sequence", "alternative_icu_morality_task_config_yaml"),
        ("boundary_exclusion_sequence", "alternative_icu_morality_task_config_yaml"),
        ("death_before_discharge_same_time_sequence", "alternative_icu_morality_task_config_yaml"),
        ("multiple_sequences_death", "alternative_icu_morality_task_config_yaml"),
        ("hematocrit_normal_sequence", "hematocrit_task_config_yaml"),
        ("hematocrit_abnormal_sequence", "hematocrit_task_config_yaml"),
        ("hematocrit_boundary_sequence", "hematocrit_task_config_yaml"),
        ("hematocrit_threshold_sequence", "hematocrit_task_config_yaml"),
    ],
)
def test_sequence_labeler(metadata_df, sequence_fixture_name, task_yaml_name, request):
    """Test ICU mortality task with different sequence patterns."""
    # Get sequence data from fixture
    time_scale = "Y"
    sequence_data = request.getfixturevalue(sequence_fixture_name)
    task_config_yaml = request.getfixturevalue(task_yaml_name)
    sequence = sequence_data["sequence"]
    expected_statuses = sequence_data["expected_statuses"]

    # Convert sequence times to the target scale
    sequence = convert_sequence_times(sequence, time_scale)
    batch_size = len(sequence)

    # Create labeler with time scale
    labeler = SequenceLabeler.from_yaml_str(
        task_config_yaml, metadata_df, batch_size=batch_size, time_scale=time_scale
    )
    gap_years = labeler.gap_days / 365

    # Set up model
    model = SimpleGenerativeModel(sequence)
    prompts = torch.zeros((batch_size), dtype=torch.long)
    print_window_tree_with_state(labeler.tree.root)

    # Process each step
    for step, expected_status in enumerate(expected_statuses):
        next_tokens, next_times, numeric_values = model.generate_next_token(prompts)
        status = labeler.process_step(next_tokens, next_times - gap_years, numeric_values).clone()
        # TODO: We currently don't check whether it correctly tracks being in WindowStatus 0 or 1
        print_window_tree_with_state(labeler.tree.root)
        status[status == 0] = 1
        expected_status[expected_status == 0] = 1

        assert torch.equal(status, expected_status), (
            f"{sequence_fixture_name} ({time_scale}) - Step {step}: "
            f"Expected status {expected_status}, got {status}"
        )

        if not model.is_finished():
            prompts = next_tokens

    # Check final labels
    labels = labeler.get_labels()
    expected_labels = (
        torch.tensor(sequence_data["label"])
        if isinstance(sequence_data["label"], list)
        else torch.tensor([sequence_data["label"]])
    )

    assert torch.equal(
        labels & (status == torch.tensor(2)), expected_labels
    ), f"{sequence_fixture_name} ({time_scale}): Expected labels {expected_labels}, got {labels}"


def test_alternative_gap_window_reference_issue(metadata_df):
    """Test handling of gap window with direct trigger reference."""
    task_config_yaml = """
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
        start: trigger  # Direct reference to trigger instead of input.end
        end: start + 72h
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

    # Setup test sequence that should trigger the issue
    test_sequence = [
        (5, 0.0, 0.0),  # Other event at index
        (5, 20.0, 0.0),  # Other event during input
        (5, 40.0, 0.0),  # Other event during gap
        (4, 72.0, 0.0),  # Death after gap
        (5, 74.0, 0.0),  # Other event after death
    ]
    expected_statuses = [
        torch.tensor([1]),  # Initial state
        torch.tensor([1]),  # Input window active
        torch.tensor([1]),  # Gap window active
        torch.tensor([1]),  # Satisfied (death)
        torch.tensor([2]),  # Remains satisfied
    ]
    expected_label = True

    # Create config from YAML and attempt to create labeler
    labeler = SequenceLabeler.from_yaml_str(task_config_yaml, metadata_df, batch_size=1, time_scale="D")

    # Set up model and test sequence
    model = SimpleGenerativeModel([test_sequence])
    prompts = torch.zeros((1), dtype=torch.long)

    # Process each step
    for step, expected_status in enumerate(expected_statuses):
        next_tokens, next_times, numeric_values = model.generate_next_token(prompts)
        status = labeler.process_step(next_tokens, next_times, numeric_values)

        assert torch.equal(
            status, expected_status
        ), f"Step {step}: Expected status {expected_status}, got {status}"

        if not model.is_finished():
            prompts = next_tokens

    # Check final labels
    labels = labeler.get_labels()
    assert expected_label == any(labels), f"Expected label {expected_label}, got {labels}"
