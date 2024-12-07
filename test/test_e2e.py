import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from czsl.labeler import create_zero_shot_task


class DummyTransformerWrapper:
    """Mock transformer that generates predictable sequences for testing."""

    def __init__(self, vocab_size: int = 5, max_seq_len: int = 100):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def forward(self, x, **kwargs):
        """Always predict uniform distribution over vocab."""
        b, t = x.shape
        # Return logits and cache
        return torch.ones(b, t, self.vocab_size), None


def generate_with_task(model, prompts, task_config, temperature=1.0, **kwargs):
    """
    Dummy generation function for testing. Simulates generation by extending prompts
    with random tokens until stopping criteria are met.

    Args:
        model: The DummyTransformerWrapper model
        prompts: Input token sequences [batch_size, seq_len]
        task_config: ZeroShotTaskConfig object
        temperature: Sampling temperature (not used in dummy implementation)
        **kwargs: Additional arguments (not used in dummy implementation)

    Returns:
        tuple: (generated_sequences, sequence_lengths)
    """
    import torch

    batch_size = prompts.shape[0]
    device = prompts.device

    # Get generation parameters from task config
    budget = task_config.get_generation_budget()
    eos_tokens = task_config.get_eos_tokens()

    # Initialize output sequences with prompts
    outputs = prompts.clone()

    # Track finished sequences
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    sequence_lengths = torch.ones(batch_size, dtype=torch.int) * outputs.shape[1]

    # Generate until all sequences are finished
    max_new_tokens = 100  # Safety limit
    for i in range(max_new_tokens):
        # Generate one random token for each unfinished sequence
        new_tokens = torch.randint(0, model.vocab_size, (batch_size, 1), device=device)
        outputs = torch.cat([outputs, new_tokens], dim=1)

        # Update finished flags based on generation budget
        if budget.budget_type == "eos_only":
            # Check for EOS tokens
            if eos_tokens:
                for b in range(batch_size):
                    if not finished[b] and new_tokens[b, 0] in eos_tokens:
                        finished[b] = True
                        sequence_lengths[b] = outputs.shape[1]

        elif budget.budget_type == "sequence_length":
            # Check sequence length
            if outputs.shape[1] >= budget.value:
                finished.fill_(True)
                sequence_lengths.fill_(outputs.shape[1])

        elif budget.budget_type == "time":
            # Simulate time budget with token count
            # For testing, assume each token takes 1 time unit
            if i >= budget.value:
                finished.fill_(True)
                sequence_lengths.fill_(outputs.shape[1])

        # Stop if all sequences are finished
        if finished.all():
            break

    return outputs, sequence_lengths


@dataclass
class GenerationTestCase:
    """Test case for generation behavior."""

    yaml_str: str
    token_map: dict
    expected_eos_tokens: list[int]
    expected_budget_type: str
    expected_budget_value: float | None = None
    expected_target_codes: list[int] = None


@pytest.fixture
def base_token_map():
    """Fixture providing basic token to code mapping."""
    return {
        0: "event_type//ICU_ADMISSION",
        1: "event_type//DEATH",
        2: "event_type//DISCHARGE",
        3: "event_type//LAB_RESULT",
        4: "event_type//MEDICATION",
    }


@pytest.fixture
def dummy_model():
    """Fixture providing a dummy transformer model."""
    return DummyTransformerWrapper(vocab_size=5, max_seq_len=20)


def create_config_file(yaml_str: str) -> Path:
    """Create a temporary YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_str)
        return Path(f.name)


@pytest.fixture
def icu_mortality_yaml():
    """Fixture providing ICU mortality prediction YAML."""
    return """
    predicates:
      icu_admission:
        code: "event_type//ICU_ADMISSION"
      death:
        code: "event_type//DEATH"
      discharge:
        code: "event_type//DISCHARGE"
      death_or_discharge:
        expr: "or(death, discharge)"
    trigger: "icu_admission"
    windows:
      observation:
        start: null
        end: "trigger + 24h"
        start_inclusive: true
        end_inclusive: true
        has:
          "_ANY_EVENT": "(1, None)"
        index_timestamp: "end"
      outcome:
        start: "observation.end"
        end: "start -> death_or_discharge"
        start_inclusive: false
        end_inclusive: true
        has: {}
        label: "death"
    """


@pytest.fixture
def lab_value_yaml():
    """Fixture providing lab value prediction YAML."""
    return """
    predicates:
      lab_normal:
        code: "event_type//LAB_RESULT"
        value_min: 0.0
        value_max: 1.0
      lab_high:
        code: "event_type//LAB_RESULT"
        value_min: 1.0
      any_lab:
        expr: "or(lab_normal, lab_high)"
    trigger: "icu_admission"
    windows:
      observation:
        start: null
        end: "trigger + 12h"
        start_inclusive: true
        end_inclusive: true
        has:
          "_ANY_EVENT": "(1, None)"
      prediction:
        start: "observation.end"
        end: "start + 24h"
        start_inclusive: false
        end_inclusive: true
        has: {}
        label: "lab_high"
    """


@pytest.fixture
def medication_yaml():
    """Fixture providing medication prediction YAML."""
    return """
    predicates:
      icu_admission:
        code: "event_type//ICU_ADMISSION"
      medication:
        code: "event_type//MEDICATION"
      lab_abnormal:
        code: "event_type//LAB_RESULT"
        value_min: 2.0
      med_trigger:
        expr: "and(lab_abnormal, icu_admission)"
    trigger: "med_trigger"
    windows:
      observation:
        start: null
        end: "trigger + 6h"
        start_inclusive: true
        end_inclusive: true
        has:
          "_ANY_EVENT": "(1, None)"
      outcome:
        start: "observation.end"
        end: "start -> medication"
        start_inclusive: false
        end_inclusive: true
        has: {}
        label: "medication"
    """


def test_generation_case(
    yaml_str: str,
    token_map: dict,
    expected_eos_tokens: list[int],
    expected_budget_type: str,
    expected_budget_value: float | None,
    expected_target_codes: list[int],
    dummy_model: DummyTransformerWrapper,
):
    """Test a single generation case."""
    config_path = create_config_file(yaml_str)
    try:
        task_config = create_zero_shot_task(yaml_path=config_path, token_to_code_map=token_map)

        # Check EOS tokens
        eos_tokens = task_config.get_eos_tokens()
        assert sorted(eos_tokens) == sorted(expected_eos_tokens)

        # Check budget
        budget = task_config.get_generation_budget()
        assert budget.budget_type.value == expected_budget_type

        if expected_budget_value is not None:
            assert budget.value == expected_budget_value

        # Check labeler target codes if specified
        if expected_target_codes is not None:
            labeler = task_config.get_task_labeler()
            assert sorted(labeler.target_codes) == sorted(expected_target_codes)

        # Test actual generation
        B, S = 2, 5  # batch_size, sequence_length
        prompts = torch.randint(0, dummy_model.vocab_size, (B, S))

        outputs, lengths = generate_with_task(
            model=dummy_model, prompts=prompts, task_config=task_config, temperature=1.0
        )

        # Verify outputs have valid shape and content
        assert outputs.shape[1] >= S
        assert torch.all(outputs >= 0)
        assert torch.all(outputs < dummy_model.vocab_size)

        # For EOS-based generation, verify it stops at EOS
        if expected_eos_tokens:
            for seq in outputs:
                # Find first EOS token
                eos_positions = [i for i, t in enumerate(seq) if t in expected_eos_tokens]
                if eos_positions:
                    first_eos = min(eos_positions)
                    # Verify no generation after first EOS
                    assert len(seq) == first_eos + 1, "Generation continued after EOS token"

    finally:
        config_path.unlink()  # Cleanup temp file


def test_icu_mortality(dummy_model, base_token_map, icu_mortality_yaml):
    """Test ICU mortality prediction task."""
    test_generation_case(
        yaml_str=icu_mortality_yaml,
        token_map=base_token_map,
        expected_eos_tokens=[1, 2],  # death or discharge
        expected_budget_type="eos_only",
        expected_budget_value=None,
        expected_target_codes=[1],  # death only
        dummy_model=dummy_model,
    )


def test_lab_value_prediction(dummy_model, base_token_map, lab_value_yaml):
    """Test lab value prediction with time window."""
    test_generation_case(
        yaml_str=lab_value_yaml,
        token_map=base_token_map,
        expected_eos_tokens=[],  # No EOS tokens, time-based
        expected_budget_type="time",
        expected_budget_value=24.0,
        expected_target_codes=[3],  # lab_result token
        dummy_model=dummy_model,
    )


def test_medication_prediction(dummy_model, base_token_map, medication_yaml):
    """Test medication prediction with complex derived predicates."""
    test_generation_case(
        yaml_str=medication_yaml,
        token_map=base_token_map,
        expected_eos_tokens=[4],  # medication token
        expected_budget_type="eos_only",
        expected_budget_value=None,
        expected_target_codes=[4],  # medication token
        dummy_model=dummy_model,
    )
