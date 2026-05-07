# Satellite Image Coordinate Prediction

A PyTorch convolutional neural network that predicts geographic coordinates (latitude/longitude) from [NASA EPIC](https://epic.gsfc.nasa.gov/) satellite imagery of Earth. The model ingests raw Earth images from the DSCOVR satellite and regresses the geographic position (lon, lat) of the sub-satellite point using a fully-convolutional backbone with a regression head.

## Features

- **Data pipeline**: Async aiohttp downloader with semaphore-limited concurrency, atomic `.tmp` → rename writes, and automatic metadata caching
- **Temporal leakage prevention**: Train/val/test splits are performed at the **date level**, ensuring no images from the same date appear in multiple splits
- **Reproducibility**: Seeded Python `random`, NumPy, and PyTorch RNGs with deterministic cuDNN flags (`torch.backends.cudnn.deterministic=True`)
- **Config validation**: Enforces consistency between `grayscale` mode and `input_channels` (1 vs 3) at configuration time
- **MPS-aware dataloaders**: Apple Silicon MPS backend automatically uses `num_workers=0` and `pin_memory=False` to avoid multiprocessing issues
- **NaN-safe Haversine**: Distance computation clamps `a ≤ 1.0` before `asin()` to prevent floating-point rounding NaNs
- **Map backends**: Cartopy (default, modern) with Basemap (legacy) fallback for world map visualizations
- **Multiple output formats**: Evaluation reports in JSON, Markdown, and CSV via `EvaluationReporter`
- **TensorBoard integration**: Training curves, model graph, hyperparameter comparison, and embedding projector
- **Learned representation visualization**: `visualize` subcommand renders KMeans clusters of the model's 128-d test embeddings on a world map — contiguous geographic regions indicate the model has learned geography in feature space

## Requirements

```
torch torchvision pandas matplotlib requests pillow tqdm numpy tensorboard aiohttp certifi scipy scikit-learn
```

Optional: `cartopy` (preferred world map rendering), `basemap` (legacy fallback), `seaborn`, `psutil`, `torchinfo`

## Setup

```bash
# Install core dependencies
pip install torch torchvision pandas matplotlib requests pillow tqdm numpy tensorboard aiohttp certifi scipy scikit-learn

# Optional: install cartopy for world map visualizations
pip install cartopy

# The NASA EPIC API does not require an API key
# (optional: export NASA_EPIC_API_KEY="your_key")
```

## Usage

```bash
# Setup data pipeline (download metadata + coordinate visualizations)
python main.py setup

# Download recent satellite images
python main.py download 7              # last 7 days
python main.py download 30             # last 30 days

# Train the regressor
python main.py train regressor
python main.py train regressor --epochs 50

# Evaluate a trained model
python main.py evaluate models/regressor_final.pth

# Visualize the learned world representation
python main.py visualize models/regressor_final.pth
python main.py visualize models/regressor_final.pth --n-clusters 12

# Run exploratory data analysis
python eda.py

# Test single prediction
python test_predictions.py --model_path models/regressor_final.pth

# Test multiple samples
python test_predictions.py --model_path models/regressor_final.pth --num_samples 6

# World map with predictions colored by error
python test_predictions.py --model_path models/regressor_final.pth --num_samples 100 --world_map
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--epochs N` | Override training epochs |
| `--batch-size N` | Override batch size |
| `--lr RATE` | Override learning rate |
| `--device DEVICE` | Force device: `auto` (default), `cuda`, `mps`, `cpu` |
| `--no-tensorboard` | Disable TensorBoard auto-launch |
| `--n-clusters N` | Number of KMeans clusters for `visualize` (default: 8) |
| `--config FILE` | Load config from JSON file |

Device auto-detection: **CUDA > MPS (Apple Silicon) > CPU**. Override with `--device`.

## Project Structure

```
main.py                 CLI entry point (setup / train / evaluate / visualize / download)
config.py               Dataclass-based configuration (DataConfig, ModelConfig, TrainingConfig)
models.py               LocationRegressor CNN with configurable conv channels
training.py             UnifiedTrainer + LocationRegressorTrainer with TensorBoard logging
data.py                 NASA EPIC API client + async image downloader
datasets.py             PyTorch Dataset, DataLoaders, CoordinateNormalizer
eda.py                  EDA with PCA, K-Means, statistics, and overview figure
visualization.py        Plotting (distributions, world maps, training curves, error maps)
evaluation_reporter.py  Comprehensive metrics + report generation (JSON/MD/CSV)
tensorboard_utils.py    TensorBoard start/stop utilities with fallback methods
test_predictions.py     Prediction testing + world error map
visualize_representation.py  Renders KMeans clusters of 128-d test embeddings on a world map
cross_validate.sh       Grid search over batch size and learning rate
```

## Data Organization

```
images/YYYY-MM-DD/*.png     Satellite images organized by date directory
combined/YYYY-MM-DD.json    Metadata per date (centroid_coordinates, etc.)
models/                     Saved model checkpoints
logs/tensorboard/           TensorBoard run logs (training + EDA)
outputs/                    Evaluation reports, figures, and CSVs
raw_data/                   Raw API response cache
```

## Model Architecture

The `LocationRegressor` CNN (`models.py`):

| Layer | Configuration |
|-------|---------------|
| Conv blocks | 3 sequential blocks: Conv2d → MaxPool2d → Tanh |
| Conv channels | [64, 128, 256] |
| Kernel | 3×3, padding=1 |
| Pool | 4×4 MaxPool |
| Hidden dim | 128 (with ReLU + Dropout 0.2) |
| Output | 2 neurons (lon, lat) |
| Weight init | Kaiming normal (conv), Xavier uniform (linear) |

The model also exposes `get_embeddings()` for feature extraction (activations before the final linear layer), logged to TensorBoard via the projector.

## Training

The `UnifiedTrainer` (`training.py`) provides:

- **Optimizers**: Adam (default), SGD, AdamW
- **Loss functions**: MSE (default), L1, Smooth L1
- **Schedulers**: StepLR (default), CosineAnnealing, ReduceLROnPlateau, None
- **Features**: Gradient clipping, checkpointing (best model by validation loss), TensorBoard logging (scalars, graph, hyperparameters, embeddings)
- **Coordinate evaluation**: Mean/median error in degrees + Haversine distance in km

## Evaluation

```bash
python main.py evaluate models/regressor_final.pth
```

Computes:

- Mean/median coordinate error (degrees) with percentile breakdown
- Mean/median Haversine distance (km)
- Accuracy thresholds (% within 1 km, 10 km, 100 km, 1000 km)

Reports are saved to `outputs/` in JSON, Markdown, and CSV formats via `EvaluationReporter`.

## Configuration

Default config is defined via dataclasses in `config.py`. Override via CLI flags or a JSON file:

```json
{
  "data": {"image_size": 64, "grayscale": false},
  "model": {"conv_channels": [64, 128, 256], "hidden_dim": 128},
  "training": {"batch_size": 32, "learning_rate": 0.001, "epochs": 100}
}
```

Config validation in `Config.__post_init__()` enforces that `grayscale=True` requires `input_channels=1` and `grayscale=False` requires `input_channels=3`.

## Reproducibility

The `_set_seed()` function in `main.py` seeds Python `random`, NumPy, and PyTorch RNGs. When CUDA is available, `torch.backends.cudnn.deterministic` is set to `True` and `benchmark` to `False`. The random seed defaults to `42` and is configurable in `TrainingConfig`.

## Temporal Leakage Prevention

`SatelliteImageDataset._split_data()` shuffles **dates** (not individual samples), then assigns entire dates to train/val/test splits. This guarantees that no images captured on the same date appear across different splits, eliminating a common source of overfitting in time-series satellite data.

## Map Visualizations

World map plots use **cartopy** by default (modern, well-maintained). If cartopy is not installed, the code falls back to **Basemap** (legacy, deprecated). If neither is available, map plots are skipped with a warning.

## Learned Representation Visualization

```bash
python main.py visualize models/regressor_final.pth
python main.py visualize models/regressor_final.pth --n-clusters 12
```

Extracts 128-d embeddings from the test set via `LocationRegressor.get_embeddings()`, runs `StandardScaler + KMeans` on them, then plots each test sample on a Miller-projection world map at its true (lon, lat), colored by its feature-space cluster id. Cluster ids are reordered by each centroid's PC1 score so adjacent ids correspond to adjacent feature-space directions, giving the categorical colormap continuity. If the model has learned geography, clusters form contiguous regions (continents, ocean basins, latitude bands); otherwise the colors look like salt-and-pepper noise. Output is saved to `outputs/representation_<timestamp>.png` at 300 dpi. Falls back to a plain matplotlib scatter when neither cartopy nor Basemap is installed.

## Exploratory Data Analysis

Run `python eda.py` to generate a comprehensive 5×4 overview figure (`outputs/eda_overview.png`):

| Row | Content |
|-----|---------|
| 1 | Mean image, pixel std dev, sample images with coordinates |
| 2 | Pixel intensity histogram, per-image brightness, lon/lat distributions |
| 3 | Coordinate coverage scatter, PCA colored by lon/lat, explained variance |
| 4 | Brightness vs lon/lat correlations, summary statistics tables |
| 5 | K-Means elbow & silhouette curves, PCA by cluster, geographic clusters, cluster center images |

Additionally computes K-Means clustering (K=2..8) with optimal K selected via silhouette score, and logs all metrics to TensorBoard under `logs/tensorboard/eda_*/`.

## TensorBoard

Training and EDA logs are saved to `logs/tensorboard/`. Training logs include:
- Loss curves (train/val)
- Learning rate
- Model computational graph
- Hyperparameter comparison
- Embedding projector (hidden-layer features colored by lon/lat)

```bash
tensorboard --logdir logs/tensorboard
python tensorboard_utils.py start    # auto-launch with fallback methods
python tensorboard_utils.py stop     # stop TensorBoard on port 6006
```

## Cross-Validation

A shell script runs a grid search over batch sizes (16, 32, 64) and learning rates (0.01, 0.001, 0.0001):

```bash
bash cross_validate.sh
```

Results are logged to `cross_validation_results.csv`.
