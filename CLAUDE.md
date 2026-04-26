# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyTorch CNN that predicts geographic coordinates (lon, lat) from NASA EPIC satellite images.

**Model**: `LocationRegressor` — 3 Conv2d blocks ([64, 128, 256] channels) → tanh → MaxPool4x4 → FC(128) → dropout → 2 outputs. Weights: Kaiming normal (conv), Xavier uniform (linear). Exposes `get_embeddings()` for feature extraction.

**Data flow**: NASA EPIC API → async download → `combined/YYYY-MM-DD.json` (metadata) + `images/YYYY-MM-DD/*.png` → `SatelliteImageDataset` → `DataLoader` → model. Coordinates stored as [lon, lat] tensors.

**Coordinate normalizer**: `CoordinateNormalizer` in `datasets.py` normalizes coords to [0,1] and denormalizes back. Provides longitude wraparound handling (`compute_longitude_error`) and Haversine distance (`compute_haversine_distance`).

**Training**: `UnifiedTrainer` → `LocationRegressorTrainer`. Supports Adam/SGD/AdamW, MSE/L1/SmoothL1, StepLR/Cosine/Plateau schedulers, gradient clipping, checkpointing (saves best model by val loss).

**Evaluation**: `EvaluationReporter` generates JSON, MD, and CSV reports. Metrics: coordinate error (degrees), Haversine distance (km), accuracy thresholds (1/10/100/1000 km).

**Device**: Auto-detected (CUDA > MPS > CPU). MPS uses `num_workers=0` due to multiprocessing limitations.

## Key Architecture Details

- `config.py`: Dataclass-based (`Config` → `DataConfig`, `ModelConfig`, `TrainingConfig`). Loaded via `Config()` or `Config.from_dict()`. CLI args override fields after construction.
- `data.py`: `EPICDataDownloader` (async image downloads with aiohttp, semaphore-limited concurrency) + `CoordinateExtractor` (parses `centroid_coordinates` from JSON metadata).
- `datasets.py`: `SatelliteImageDataset` loads images, splits train/val/test (80/10/10), applies transforms. `create_dataloaders()` auto-tunes `num_workers`/`pin_memory` per device type.
- `main.py`: CLI entry point — commands: `setup`, `train regressor`, `evaluate <path>`, `download [days]`.
- `training.py`: Base `UnifiedTrainer` with TensorBoard logging (scalars, graph, embeddings projector, hyperparameters). `LocationRegressorTrainer` subclass adds coordinate evaluation.
- `eda.py`: Standalone (run directly). Computes image statistics, PCA (2 components), K-Means (K=2..8, optimal via silhouette), correlations. Generates 5×4 overview figure and TensorBoard logs.
- `test_predictions.py`: Standalone. Loads model, runs on test set, plots single/multiple/world-error-map visualizations with Basemap.

## Commands

```bash
# Train
python main.py train regressor
python main.py train regressor --epochs 100 --batch-size 32 --lr 0.001

# Evaluate
python main.py evaluate models/regressor_final.pth

# Download
python main.py download 7

# Setup data pipeline
python main.py setup

# EDA
python eda.py

# Test predictions
python test_predictions.py --model_path models/regressor_final.pth --num_samples 1 --show
python test_predictions.py --model_path models/regressor_final.pth --num_samples 100 --world_map

# TensorBoard
tensorboard --logdir logs/tensorboard
python tensorboard_utils.py start
python tensorboard_utils.py stop

# Override config
python main.py train regressor --config my_config.json
python main.py train regressor --device mps --no-tensorboard
```
