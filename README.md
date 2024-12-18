# Clinical Zero-Shot Labeler

A tool for adapting [ACES (Automated Cohort and Event Selection)](https://github.com/justin13601/ACES/tree/main) task schemas to zero-shot labeling of clinical sequences.

## Overview

The Clinical Zero-Shot Labeler extends ACES task schemas, originally designed for cohort extraction and binary classification tasks, to work with generative models. This allows you to:

1. Use existing ACES task definitions for generative tasks
2. Control sequence generation using ACES predicates and windows
3. Extract labels from generated sequences using ACES criteria

By leveraging the ACES schema, you can define complex clinical tasks like:

- ICU mortality prediction
- Lab value forecasting
- Treatment response prediction
- Readmission risk assessment

All without needing to modify code or retrain models, and maintaining compatibility with existing ACES configurations.

## About ACES

ACES (Automated Cohort and Event Selection) is a framework for defining clinical tasks through a structured YAML schema. Originally designed for:

- Cohort extraction from clinical data
- Binary classification task definition
- Temporal relationship specification
- Event sequence validation

This library extends ACES to work with generative models by:

- Converting ACES predicates into generation stopping criteria
- Using ACES windows to control sequence length and timing
- Applying ACES labeling logic to generated sequences

## Installation

```bash
git clone git@github.com:Oufattole/clinical-zeroshot-labeler.git
cd clinical-zeroshot-labeler
pip install -e .[tests,dev]
```

## Quick Start

1. Define your task in a YAML file:

```yaml
predicates:
  icu_admission:
    code: event_type//ICU_ADMISSION
  death:
    code: event_type//DEATH
  discharge:
    code: event_type//DISCHARGE
  death_or_discharge:
    expr: or(death, discharge)

trigger: icu_admission

windows:
  observation:
    start:
    end: trigger + 24h
    start_inclusive: true
    end_inclusive: true
    has:
      _ANY_EVENT: (1, None)
    index_timestamp: end

  outcome:
    start: observation.end
    end: start -> death_or_discharge
    start_inclusive: false
    end_inclusive: true
    has: {}
    label: death
```

2. Create task configuration:

```python
from clinical_zeroshot_labeler import create_zero_shot_task

# Map token IDs to predicate codes
token_map = {
    0: "event_type//ICU_ADMISSION",
    1: "event_type//DEATH",
    2: "event_type//DISCHARGE",
}

# Create task config
task_config = create_zero_shot_task(
    yaml_path="icu_mortality.yaml", token_to_code_map=token_map
)
```

3. Generate sequences and get labels:

```python
# Generate sequences
outputs, lengths = generate_with_task(
    model=model, prompts=prompts, task_config=task_config, temperature=0.7
)

# Get labels
labeler = task_config.get_task_labeler()
labels, unknown = labeler(trajectory_batch)
```

## Task Configuration

ACES Tasks are defined through YAML files with two main components:

### Predicates

Define events and conditions to match:

```yaml
predicates:
  # Simple event
  icu_admission:
    code: event_type//ICU_ADMISSION

  # Event with numeric criteria
  high_glucose:
    code: GLUCOSE
    value_min: 180
    value_max:

  # Composite predicate
  any_critical:
    expr: or(high_glucose, low_bp)
```

### Windows

Define temporal relationships and constraints:

```yaml
windows:
  # Observation window
  observation:
    start:       # Start of record
    end: trigger + 24h
    start_inclusive: true
    end_inclusive: true
    has:
      _ANY_EVENT: (1, None)

  # Prediction window
  outcome:
    start: observation.end
    end: start -> death_or_discharge
    start_inclusive: false
    end_inclusive: true
    has: {}
    label: death
```

## Generation Control

The configuration automatically handles:

- EOS tokens based on ACES task predicates in the prediction window
  - Edge cases such as `end_inclusive` are also handled.
- Time-based stopping based on the ACES task predicates and prediction window end time
- Sequence length limits, that are separately added by users. If a sequence length limit
  is imposed and generation does not complete by that sequence limit, an unknown label is returned.
