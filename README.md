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
- Readmission risk assessment
- etc.

All without needing to modify code or retrain models, and maintaining compatibility with existing ACES configurations.

## Installation

```bash
pip install clinical-zeroshot-labeler
```

## Quick Start

1. Define your task in YAML:

```yaml
predicates:
  hospital_discharge:
    code: {regex: HOSPITAL_DISCHARGE//.*}
  lab:
    code: {regex: LAB//.*}
  abnormal_lab:
    code: {regex: LAB//.*}
    value_min: 2.0
    value_min_inclusive: true

trigger: hospital_discharge

windows:
  input:
    start:
    end: trigger
    start_inclusive: true
    end_inclusive: true
    index_timestamp: end
  target:
    start: input.end
    end: start + 4d
    start_inclusive: false
    end_inclusive: true
    has:
      lab: (1, None)
    label: abnormal_lab
```

2. Set up your metadata mapping:

```python
import polars as pl

# Load a metadata mapping of medical codes to vocabulary indices your generative model generates
metadata_df = pl.DataFrame(
    {
        "code": [
            "PAD",
            "HOSPITAL_DISCHARGE//MEDICAL",
            "LAB//NORMAL",
            "LAB//HIGH",
        ]
    }
).with_row_index("code/vocab_index")
```

3. Process sequences and get labels:

```python
from clinical_zeroshot_labeler import SequenceLabeler

# Initialize labeler
labeler = SequenceLabeler.from_yaml_str(task_config_yaml, metadata_df, batch_size=2)

# Process tokens one at a time
while not labeler.is_finished():
    # Get next tokens from your model
    tokens, times, values = model.generate_next_token(prompts)

    # Update labeler state
    status = labeler.process_step(tokens, times, values)
    print(
        f"Status: {status}"
    )  # Shows 0=Undetermined, 1=Active, 2=Satisfied, 3=Impossible

    # Update your model's prompts as needed
    prompts = tokens

# Get final labels
labels = labeler.get_labels()
```

See the `notebooks/tutorial.ipynb` to run the SequenceLabeler on a mocked Generator.

## API Reference

### SequenceLabeler

Main class for processing sequences and extracting labels:

```python
# Create from YAML string
labeler = SequenceLabeler.from_yaml_str(yaml_str, metadata_df, batch_size)

# Create from YAML file
labeler = SequenceLabeler.from_yaml_file(yaml_path, metadata_df, batch_size)

# Process tokens (returns status tensor)
status = labeler.process_step(tokens, times, values)

# Check if finished
is_done = labeler.is_finished()

# Get labels
labels = labeler.get_labels()
```

The labeler tracks window states for each sequence:

- `0`: Undetermined - Initial state
- `1`: Active - Currently processing
- `2`: Satisfied - Success/completion
- `3`: Impossible - Failed/invalid
