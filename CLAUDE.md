# CLAUDE.md — fantastic-palm-tree

Guidance for AI coding agents working on this repository.

## Project Overview

PyTorch CNN (`LocationRegressor`) that predicts geographic coordinates (lon, lat) from NASA EPIC satellite imagery of Earth. The model uses 3 Conv2d → Tanh → MaxPool blocks, a 128-dim hidden layer, and 2 output neurons. Data flows from the NASA EPIC API through async download into per-date JSON metadata + PNG images, then into a `SatelliteImageDataset` with date-level train/val/test splitting to prevent temporal leakage.

## Architecture Summary

| File | Role |
|------|------|
| `main.py` | CLI entry point: `setup`, `train regressor`, `evaluate <path>`, `visualize <path>`, `download [days]`. Sets seeds for reproducibility. |
| `config.py` | Dataclass-based `Config(DataConfig, ModelConfig, TrainingConfig)`. Auto-detects device (CUDA > MPS > CPU). Validates grayscale ↔ input_channels consistency in `__post_init__`. |
| `data.py` | `EPICDataDownloader` (async aiohttp, semaphore-limited concurrency, atomic .tmp → rename). `CoordinateExtractor` parses `centroid_coordinates` from per-date JSON. |
| `datasets.py` | `SatelliteImageDataset` loads images + coords. `_split_data()` shuffles **dates** (not samples) to prevent temporal leakage. `CoordinateNormalizer` normalizes to [0,1], denormalizes, computes Haversine distance (with NaN clamping) and longitude wraparound error. `create_dataloaders()` adjusts `num_workers`/`pin_memory` per device (MPS → workers=0, pin_memory=False). |
| `models.py` | `LocationRegressor(config.model)` — 3 conv blocks [64,128,256] → Tanh → MaxPool4 → FC(128) → Dropout(0.2) → 2 outputs. Kaiming normal (conv), Xavier uniform (linear). Exposes `get_embeddings()` for TensorBoard projector. |
| `training.py` | `UnifiedTrainer` / `LocationRegressorTrainer`. TensorBoard logging (scalars, graph, hparams, embeddings). Gradient clipping, schedulers (StepLR/Cosine/Plateau), best-model checkpointing. |
| `evaluation_reporter.py` | Comprehensive eval → JSON + Markdown + CSV reports. Metrics: coordinate error (deg), Haversine distance (km), accuracy thresholds. |
| `visualization.py` | Map backend: **cartopy** (preferred) → Basemap (fallback). Distribution plots, training curves, prediction scatter, world error maps. |
| `eda.py` | Standalone EDA: image statistics, PCA (2 components), K-Means (K=2..8, optimal via silhouette), correlations, 5×4 overview figure, TensorBoard logging. |
| `test_predictions.py` | Single/multiple/world-error-map prediction visualization with cartopy/Basemap backend. |
| `visualize_representation.py` | Extracts 128-d test embeddings via `model.get_embeddings()`, runs `StandardScaler + KMeans` on them, reorders cluster ids by each centroid's PC1 score for colormap continuity, and renders the clusters on a Miller-projection world map. Reuses `_draw_miller_map` / `load_model` from `test_predictions.py`; falls back to a plain matplotlib scatter when neither cartopy nor Basemap is installed. |
| `tensorboard_utils.py` | Start/stop TensorBoard with multiple fallback methods. |

## Key Design Decisions & Gotchas

- **Temporal leakage prevention**: `SatelliteImageDataset._split_data()` shuffles dates, then assigns all samples from each date to train/val/test. This ensures no same-date samples appear across splits.
- **NaN clamping in Haversine**: `CoordinateNormalizer.compute_haversine_distance()` clamps `a = torch.clamp(a, max=1.0)` before `asin()` to prevent tiny float rounding errors from producing NaN.
- **MPS quirks**: Apple Silicon MPS backend requires `num_workers=0` and `pin_memory=False` in DataLoader because MPS doesn't support multiprocessing shared memory. Handled in `create_dataloaders()`.
- **Atomic downloads**: `EPICDataDownloader` writes images to `*.tmp` then renames atomically, preventing corrupt partial files.
- **Config validation**: `Config.__post_init__()` enforces `grayscale=True → input_channels=1` and `grayscale=False → input_channels=3`, raising `ValueError` on mismatch.
- **Reproducibility**: `main._set_seed(seed)` seeds Python `random`, `numpy`, `torch`, and sets `torch.backends.cudnn.deterministic=True`, `benchmark=False`.
- **Device auto-detection**: `Config._get_best_device()`: CUDA > MPS > CPU. Overridable via `--device` flag.
- **Weights-only loading**: `main.evaluate_model_performance()` handles PyTorch 2.6+ `weights_only` security restriction with fallback to `add_safe_globals`.

## Build / Run Commands

```bash
# Setup data pipeline (download metadata + coordinate visualizations)
python main.py setup

# Download recent images (last N days)
python main.py download 7
python main.py download 30

# Train the regressor
python main.py train regressor
python main.py train regressor --epochs 50 --batch-size 32 --lr 0.001

# Evaluate a trained model
python main.py evaluate models/regressor_final.pth

# Visualize the learned world representation (KMeans on 128-d embeddings, plotted on a world map)
python main.py visualize models/regressor_final.pth
python main.py visualize models/regressor_final.pth --n-clusters 12

# Override config from JSON file
python main.py train regressor --config my_config.json

# Force device
python main.py train regressor --device mps --no-tensorboard

# Exploratory data analysis
python eda.py

# Test predictions
python test_predictions.py --model_path models/regressor_final.pth
python test_predictions.py --model_path models/regressor_final.pth --num_samples 6
python test_predictions.py --model_path models/regressor_final.pth --num_samples 100 --world_map

# TensorBoard
tensorboard --logdir logs/tensorboard
python tensorboard_utils.py start
python tensorboard_utils.py stop

# Cross-validation grid search
bash cross_validate.sh
```

## Development Conventions

- **Testing**: No formal test framework. Validate changes by training on a small epoch count (`--epochs 5`) and checking evaluation metrics.
- **Formatting**: Standard Python with docstrings on all public classes/functions. Use `logging.getLogger(__name__)` for logging.
- **Commit style**: One-line descriptive messages. Recent example: `"Fix 11 bugs: data correctness, reproducibility, and map backend"`.
- **Imports**: Standard lib → third-party → local, separated by blank lines.
- **Device handling**: Always check `device.type == 'mps'` or `torch.cuda.is_available()` when making hardware-dependent decisions.

## Dependencies

Core: `torch torchvision pandas matplotlib requests pillow tqdm numpy tensorboard aiohttp certifi scipy scikit-learn`
Optional: `cartopy` (world map plots, preferred), `basemap` (fallback), `seaborn`, `psutil`, `torchinfo`
