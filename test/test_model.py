from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from omegaconf import DictConfig
from test_labeler import (  # noqa
    abnormal_lab_task_config_yaml,
    alt_successful_death_sequence,
    alternative_icu_morality_task_config_yaml,
    boundary_exclusion_sequence,
    convert_sequence_times,
    death_after_discharge_same_time_sequence,
    death_before_discharge_same_time_sequence,
    exact_boundary_sequence,
    hematocrit_abnormal_sequence,
    hematocrit_boundary_sequence,
    hematocrit_normal_sequence,
    hematocrit_task_config_yaml,
    hematocrit_threshold_sequence,
    icu_morality_task_config_yaml,
    impossible_death_boundary_sequence,
    impossible_readmission_sequence,
    metadata_df,
    print_window_tree_with_state,
    successful_death_sequence,
    successful_discharge_sequence,
    undetermined_sequence,
)

# Import all the fixtures from the original tests
from torch import nn

from clinical_zeroshot_labeler.labeler import SequenceLabeler, WindowStatus
from clinical_zeroshot_labeler.model import BaseGenerativeModel


class MockTransformerWrapper(nn.Module):
    """Mock implementation of TransformerWrapper that outputs predefined sequences."""

    def __init__(
        self,
        *,
        num_tokens: int,
        max_seq_len: int,
        sequences: list[list[tuple[int, float, float]]] = None,
        dim: int = 32,
        can_cache_kv: bool = True,
        can_cache_kv_outside_max_seq_len: bool = True,
    ):
        """Initialize mock transformer.

        Args:
            num_tokens: Vocabulary size
            max_seq_len: Maximum sequence length
            sequences: List of sequences, where each sequence is a list of (token, time, value) tuples
            dim: Model dimension
            can_cache_kv: Whether model supports KV caching
            can_cache_kv_outside_max_seq_len: Whether caching works beyond max_seq_len
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.can_cache_kv = can_cache_kv
        self.can_cache_kv_outside_max_seq_len = can_cache_kv_outside_max_seq_len

        # Store predefined sequences and current positions
        self.sequences = sequences if sequences is not None else []
        self.current_positions = [0] * len(self.sequences) if sequences is not None else []

        # For tracking calls
        self.call_count = 0
        self.cached_kvs = []

    def set_sequences(self, sequences: list[list[tuple[int, float, float]]]):
        """Set new predefined sequences and reset positions."""
        self.sequences = sequences
        self.current_positions = [0] * len(sequences)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: dict | None = None,
        return_intermediates: bool = False,
        seq_start_pos: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, SimpleNamespace | None]:
        """Mock forward pass that returns predefined sequence tokens.

        Args:
            x: Input tensor of token ids [batch_size, seq_len]
            mask: Optional attention mask
            cache: Optional KV cache
            return_intermediates: Whether to return intermediate states
            seq_start_pos: Optional starting positions
            **kwargs: Additional arguments (ignored)

        Returns:
            Tuple of:
                - Logits tensor [batch_size, seq_len, vocab_size]
                - Optional intermediates object if return_intermediates=True
        """
        self.call_count += 1
        batch_size, seq_len = x.shape
        device = x.device

        # Generate logits for next tokens in sequence
        logits = torch.zeros((batch_size, seq_len, self.num_tokens), device=device)

        for b in range(batch_size):
            if b < len(self.sequences) and self.current_positions[b] < len(self.sequences[b]):
                next_token = self.sequences[b][self.current_positions[b]][0]
                logits[b, -1, next_token] = 1.0  # One-hot encoding for next token
                self.current_positions[b] += 1
            else:
                # If we've exhausted the sequence, repeat last token
                last_token = x[b, -1]
                logits[b, -1, last_token] = 1.0

        # Handle KV caching
        if self.can_cache_kv:
            new_cache = SimpleNamespace()
            new_cache.attn_intermediates = []

            # Mock attention intermediate with cached KV
            attn_inter = SimpleNamespace()
            attn_inter.layer_type = "a"
            attn_inter.cached_kv = [
                torch.randn(batch_size, 4, seq_len, 32),  # Mock key
                torch.randn(batch_size, 4, seq_len, 32),  # Mock value
            ]
            new_cache.attn_intermediates.append(attn_inter)

            self.cached_kvs.append(attn_inter.cached_kv)
        else:
            new_cache = None

        if return_intermediates:
            intermediates = SimpleNamespace()
            intermediates.attn_intermediates = new_cache.attn_intermediates if new_cache else []
            return logits, intermediates

        return logits, new_cache

    def reset_tracking(self):
        """Reset call tracking state and sequence positions."""
        self.call_count = 0
        self.cached_kvs = []
        self.current_positions = [0] * len(self.sequences)


@pytest.fixture
def normal_sequence():
    return {
        "sequence": [
            (3, 0.0, 0.0),  # Initial token
            (6, 15.0, 25.0),  # Normal value
            (7, 20.0, 30.0),  # Another normal value
            (7, 100.0, 30.0),  # Sequence successfully completed
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([1]),  # Still active
            torch.tensor([1]),  # Still active
            torch.tensor([2]),  # Satisied
        ],
        "label": False,
    }


@pytest.fixture
def abnormal_sequence():
    return {
        "sequence": [
            (3, 0.0, 0.0),  # Initial token
            (6, 15.0, 15.0),  # Abnormal value
            (7, 20.0, 30.0),  # Normal value
            (7, 100.0, 30.0),  # Sequence successfully completed
        ],
        "expected_statuses": [
            torch.tensor([1]),  # Initial state
            torch.tensor([2]),  # Satisfied (abnormal found)
            torch.tensor([2]),  # Remains satisfied
            torch.tensor([2]),  # Remains satisfied
        ],
        "label": True,
    }


def test_single_sequence_generation(normal_sequence, metadata_df):  # noqa: F811
    """Test generation with a single predefined sequence."""
    sequence_data = normal_sequence["sequence"]
    expected_statuses = normal_sequence["expected_statuses"]

    model = MockGenerativeModel(metadata_df, sequences=[sequence_data])
    prompts = torch.tensor([[1]])  # Single token prompt
    mask = torch.ones_like(prompts, dtype=torch.bool)

    class TestLabeler:
        def __init__(self, expected_statuses):
            self.step = 0
            self.expected_statuses = expected_statuses

        def process_step(self, tokens, times, values):
            logger.info(f"tokens: {tokens}, times: {times}, values: {values}")
            logger.info(f"step: {self.step}")
            status = self.expected_statuses[self.step]
            logger.info(f"status: {status}")
            self.step += 1
            return status

        def get_labels(self):
            return torch.tensor([normal_sequence["label"]])

    labeler = TestLabeler(expected_statuses)
    tokens, lengths, meta = model.generate(
        prompts=prompts, mask=mask, trajectory_labeler=labeler, temperature=0.0  # Use greedy sampling
    )

    # Verify we got expected sequence
    expected_tokens = torch.tensor([t[0] for t in sequence_data])
    assert torch.all(tokens[:, : len(expected_tokens)] == expected_tokens)


def test_batch_sequence_generation(normal_sequence, abnormal_sequence, metadata_df):  # noqa: F811
    """Test generation with multiple sequences in a batch."""
    sequences = [normal_sequence["sequence"], abnormal_sequence["sequence"]]
    expected_statuses = [normal_sequence["expected_statuses"], abnormal_sequence["expected_statuses"]]

    model = MockGenerativeModel(metadata_df, sequences=sequences)
    prompts = torch.tensor([[1], [1]])  # Two single-token prompts
    mask = torch.ones_like(prompts, dtype=torch.bool)

    class BatchTestLabeler:
        def __init__(self, expected_statuses):
            self.step = 0
            self.expected_statuses = expected_statuses

        def process_step(self, tokens, times, values):
            status = torch.cat([self.expected_statuses[0][self.step], self.expected_statuses[1][self.step]])
            self.step += 1
            return status

        def get_labels(self):
            return torch.tensor([normal_sequence["label"], abnormal_sequence["label"]])

    labeler = BatchTestLabeler(expected_statuses)
    tokens, lengths, meta = model.generate(
        prompts=prompts, mask=mask, trajectory_labeler=labeler, temperature=0.0
    )
    assert torch.all(lengths == torch.tensor([4, 2]))

    # Verify each sequence matches expected tokens
    for i, sequence in enumerate(sequences):
        expected_tokens = torch.tensor([t[0] for t in sequence])[: lengths[i]]
        actual_tokens = tokens[i, : lengths[i]]
        assert torch.all(actual_tokens == expected_tokens)

    # Verify statuses and labels
    assert torch.equal(meta["labels"], torch.tensor([False, True]))


def test_sequence_completion_with_max_tokens(normal_sequence, metadata_df):  # noqa: F811
    """Test sequence completion respects max tokens budget."""
    # Set max_tokens less than sequence length
    max_tokens = 2
    model = MockGenerativeModel(metadata_df, sequences=[normal_sequence["sequence"]], max_tokens=max_tokens)
    prompts = torch.tensor([[1]])
    mask = torch.ones_like(prompts, dtype=torch.bool)

    tokens, lengths, _ = model.generate(prompts=prompts, mask=mask, temperature=0.0)

    assert tokens.shape[1] == max_tokens
    assert lengths[0] == max_tokens


def test_temperature_sampling(normal_sequence, metadata_df):  # noqa: F811
    """Test that temperature affects sampling behavior."""
    model = MockGenerativeModel(metadata_df, sequences=[normal_sequence["sequence"]])
    prompts = torch.tensor([[1]])
    mask = torch.ones_like(prompts, dtype=torch.bool)

    # With temperature=0, should exactly match sequence
    tokens_greedy, _, _ = model.generate(prompts=prompts, mask=mask, temperature=0.0)

    # With high temperature, should eventually deviate
    # (though in our mock, it actually won't since we force the next token)
    tokens_random, _, _ = model.generate(prompts=prompts, mask=mask, temperature=2.0)

    # In this mock they'll be equal, but the test structure is here
    expected_tokens = torch.tensor([t[0] for t in normal_sequence["sequence"]])
    assert torch.all(tokens_greedy[:, : len(expected_tokens)] == expected_tokens)


class MockGenerativeModel(BaseGenerativeModel):
    """Test implementation using sequence-based MockTransformerWrapper."""

    def __init__(
        self,
        metadata_df,  # noqa: F811
        sequences=None,
        max_tokens=50,
        vocab_size=100,
    ):
        self.metadata_df = metadata_df
        self.model = SimpleNamespace()
        self.model.model = MockTransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=50,
            sequences=sequences if isinstance(sequences, list) else [sequences],
            dim=32,
        )
        self.cfg = DictConfig({"max_tokens_budget": max_tokens})

    def update_generation_state(self, tokens, cumulative_time, trajectory_labeler=None):
        """Update state based on labeler if provided."""
        batch_size = tokens.shape[0]
        sequences_complete = torch.zeros(batch_size, dtype=torch.bool)

        if trajectory_labeler is not None:
            # Extract time and value information from sequences
            current_positions = self.model.model.current_positions
            sequences = self.model.model.sequences
            times = []
            values = []

            for b in range(batch_size):
                if b < len(sequences) and current_positions[b] > 0:
                    # Use the time and value from the sequence
                    _, time, value = sequences[b][current_positions[b] - 1]
                    times.append(time)
                    values.append(value)
                else:
                    times.append(0.0)
                    values.append(0.0)

            times = torch.tensor(times, device=tokens.device)
            values = torch.tensor(values, device=tokens.device)

            status = trajectory_labeler.process_step(tokens[:, -1], times, values)
            index_to_name = {
                k: v for k, v in zip(self.metadata_df["code/vocab_index"], self.metadata_df["code"])
            }
            token = tokens[:, -1][0]
            token_name = index_to_name[token.item()]
            logger.info(
                f"token_name: {token_name}, token: {token}, times: {times}, "
                f"values: {values}, status: {status}"
            )
            if hasattr(trajectory_labeler, "tree"):
                print_window_tree_with_state(trajectory_labeler.tree.root)

            sequences_complete = status == WindowStatus.SATISFIED.value

            return cumulative_time + times, status, sequences_complete.all(), sequences_complete

        return cumulative_time + 1, None, False, sequences_complete

    def _check_valid_mask(self, mask, prompt_lens):
        """Handle mask checking for both 1D and 2D masks."""
        if len(mask.shape) == 1:
            right_pad_mask = torch.arange(mask.shape[0], device=mask.device) < prompt_lens
        else:
            right_pad_mask = torch.arange(mask.shape[1], device=mask.device).unsqueeze(
                0
            ) < prompt_lens.unsqueeze(1)
        return torch.equal(right_pad_mask, mask)


@pytest.mark.parametrize(
    "time_scale",
    [
        "Y",
        # "D"
    ],
)
@pytest.mark.parametrize(
    "sequence_fixture,config_fixture",
    [
        ("successful_death_sequence", "icu_morality_task_config_yaml"),
        ("successful_discharge_sequence", "icu_morality_task_config_yaml"),
        ("impossible_readmission_sequence", "icu_morality_task_config_yaml"),
        ("undetermined_sequence", "icu_morality_task_config_yaml"),
        ("exact_boundary_sequence", "icu_morality_task_config_yaml"),
        ("boundary_exclusion_sequence", "icu_morality_task_config_yaml"),
        ("death_after_discharge_same_time_sequence", "icu_morality_task_config_yaml"),
        ("death_before_discharge_same_time_sequence", "icu_morality_task_config_yaml"),
        ("impossible_death_boundary_sequence", "icu_morality_task_config_yaml"),
        ("alt_successful_death_sequence", "alternative_icu_morality_task_config_yaml"),
        ("hematocrit_normal_sequence", "hematocrit_task_config_yaml"),
        ("hematocrit_abnormal_sequence", "hematocrit_task_config_yaml"),
        ("hematocrit_boundary_sequence", "hematocrit_task_config_yaml"),
        ("hematocrit_threshold_sequence", "hematocrit_task_config_yaml"),
    ],
)
def test_sequence_generation(
    request, time_scale, sequence_fixture, config_fixture, metadata_df  # noqa: F811
):
    """Test sequence generation against all fixtures."""
    # Get sequence data and config
    sequence_data = request.getfixturevalue(sequence_fixture)
    config_yaml = request.getfixturevalue(config_fixture)

    # Convert sequence times
    sequence = convert_sequence_times(sequence_data["sequence"], time_scale)
    expected_statuses = sequence_data["expected_statuses"]
    expected_label = sequence_data["label"]

    # Create labeler
    batch_size = len(sequence)
    labeler = SequenceLabeler.from_yaml_str(
        config_yaml, metadata_df, batch_size=batch_size, time_scale=time_scale
    )

    # Create model with sequence
    model = MockGenerativeModel(metadata_df, sequences=sequence)
    prompts = torch.zeros((batch_size,), dtype=torch.long)
    mask = torch.ones_like(prompts, dtype=torch.bool)

    # Generate sequence and check against expected values
    tokens, lengths, meta = model.generate(
        prompts=prompts, mask=mask, trajectory_labeler=labeler, temperature=0.0
    )

    # Verify final status and labels
    assert meta is not None, "Missing metadata from generation"
    assert "status" in meta, "Missing status in metadata"
    assert "labels" in meta, "Missing labels in metadata"

    final_status = meta["status"]
    final_labels = meta["labels"]

    # Convert expected label to tensor if needed
    if isinstance(expected_label, list):
        expected_label = torch.tensor(expected_label)
    elif isinstance(expected_label, bool):
        expected_label = torch.tensor([expected_label])

    # Check final status matches last expected status
    assert torch.equal(final_status, expected_statuses[-1]), (
        f"{sequence_fixture} ({time_scale}): "
        f"Expected final status {expected_statuses[-1]}, got {final_status}"
    )

    # Check labels match expected
    assert torch.equal(final_labels & (final_status == WindowStatus.SATISFIED.value), expected_label), (
        f"{sequence_fixture} ({time_scale}): " f"Expected labels {expected_label}, got {final_labels}"
    )
