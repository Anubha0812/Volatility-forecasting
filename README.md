# TimesFM for Realized Volatility Forecasting

This repository contains the implementation code for the paper **Foundation Time-Series AI Model for Realized Volatility Forecasting**. The project evaluates whether a pre-trained time-series foundation model, TimesFM, can be used for one-day-ahead realized-volatility forecasting, both in zero-shot form and after incremental fine-tuning.

The main objective is to provide a practical forecasting framework that reuses a pre-trained time-series representation and can be adapted to financial volatility data without requiring users to specify and estimate a separate econometric volatility model for each asset.

## Overview

The repository uses Google's TimesFM model for realized-volatility forecasting. The experiments include:

* Zero-shot TimesFM forecasting.
* Incremental fine-tuning of TimesFM using recent realized-volatility observations.
* Forecast evaluation using standard volatility-forecasting loss functions.
* Comparison with classical econometric benchmarks and additional neural forecasting baselines.

The incremental fine-tuning setup updates the pre-trained checkpoint sequentially as new market data become available. This makes the framework naturally compatible with online-learning applications.

## Model Checkpoint

The implementation uses the publicly available TimesFM checkpoint from Hugging Face:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    local_dir="/path/to/tfm",
    cache_dir="/path/to/cache",
    repo_id="google/timesfm-2.0-500m-jax"
)
```

The checkpoint path is then passed to the TimesFM model and updated during incremental fine-tuning.

## Key Parameters

| Parameter              | Value                             |
| ---------------------- | --------------------------------- |
| Model                  | TimesFM 2.0, 500M, JAX checkpoint |
| Checkpoint             | `google/timesfm-2.0-500m-jax`     |
| Backend                | GPU                               |
| Context length         | 512                               |
| Forecast horizon       | 1 day ahead                       |
| Random seed            | 1234                              |
| Fine-tuning framework  | JAX/PAX                           |
| Optimizer              | Adam                              |
| Learning-rate schedule | Cosine schedule                   |
| Early stopping         | Validation-loss based             |
| Fine-tuning strategy   | Main transformer layers fixed     |

## Incremental Fine-Tuning

The incremental fine-tuning procedure starts from the pre-trained TimesFM checkpoint. For each asset, the data are split into train, validation, and test segments. The model is fine-tuned on the current training window, validated on the validation segment, and evaluated on the test segment. The resulting checkpoint is then reused as the starting point for the next update.

This process allows the model to adapt to new realized-volatility observations without retraining the foundation model from scratch.

## Data

The empirical analysis uses realized-volatility data from the Oxford-Man Institute Realized Library. The raw data are not included in this repository. Users should download the data from the original source and format it according to the expected input structure.

A typical input file should contain:

* `Date`: trading date.
* `Symbol`: market index identifier.
* realized-volatility or realized-variance column, such as `rv5_ss`.

Example:

```text
Date,Symbol,rv5_ss
01/01/2000,.SPX,...
02/01/2000,.SPX,...
```

## Repository Structure

A typical structure is:

```text
.
├── datasets/
│   └── voldata.csv
├── experiments/
│   └── extended_benchmarks/
│       └── ft_code.py
├── incremental_ft/
│   └── forecast outputs
├── var_incre/
│   └── saved incremental checkpoints
├── scripts/
│   └── TimesFM inference and fine-tuning scripts
└── README.md
```

The exact folder names can be adapted depending on the local setup.

## Running the Experiments

### 1. Install dependencies

Please follow the official TimesFM installation instructions and ensure that the required GPU/JAX environment is correctly configured.

Main Python dependencies include:

```text
timesfm
jax
jaxlib
paxml
praxis
numpy
pandas
tensorflow
huggingface_hub
tqdm
matplotlib
```

### 2. Prepare the data

Place the realized-volatility dataset in the `datasets/` folder. The default script expects:

```text
datasets/voldata.csv
```

### 3. Download the TimesFM checkpoint

Update the local checkpoint paths in the script:

```python
local_dir = "/path/to/tfm"
cache_dir = "/path/to/cache"
repo_id = "google/timesfm-2.0-500m-jax"
```

### 4. Run zero-shot forecasting or incremental fine-tuning

Run the corresponding Python script for TimesFM inference or incremental fine-tuning. The incremental fine-tuning code loops over assets, updates the checkpoint sequentially, and saves forecasts and actual values for evaluation.


These files can be used to compute loss functions such as MSE, MAE, QLIKE, MAPE, sMAPE, and MDA.

## Reproducibility

The implementation fixes the random seed at:

```python
jax.random.PRNGKey(seed=1234)
```

The experiments use the GPU backend, and the fine-tuning setup keeps the main stacked transformer layers fixed while updating the remaining trainable components. Early stopping is based on validation loss with patience set to 5.

For exact replication, users should record their hardware configuration, GPU type, CUDA/JAX versions, and running times.

## Code Availability

This repository provides scripts for:

* TimesFM checkpoint loading.
* Zero-shot forecasting.
* Incremental fine-tuning.
* Forecast storage.


Please also cite the original TimesFM work and comply with the TimesFM model license.

## License

This project uses the TimesFM repository and checkpoint released by Google. Users should review and comply with the license terms of TimesFM and any third-party dependencies. The realized-volatility data should be used according to the terms of the Oxford-Man Institute Realized Library.
