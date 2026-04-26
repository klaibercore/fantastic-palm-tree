# Satellite Image Coordinate Prediction

A PyTorch CNN that predicts geographic coordinates (latitude/longitude) from [NASA EPIC](https://epic.gsfc.nasa.gov/) satellite imagery of the Earth.

## Requirements

```
torch torchvision pandas matplotlib requests pillow tqdm numpy tensorboard aiohttp certifi scipy scikit-learn
```

Optional: `basemap` (world map plots), `seaborn`, `psutil`

## Setup

```bash
pip install torch torchvision pandas matplotlib requests pillow tqdm numpy tensorboard aiohttp certifi scipy scikit-learn
```

The NASA EPIC API does not require an API key (optional: `export NASA_EPIC_API_KEY="your_key"`).

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
| `--config FILE` | Load config from JSON file |

Device auto-detection: **CUDA > MPS (Apple Silicon) > CPU**

## Project Structure

```
main.py                 CLI entry point (setup / train / evaluate / download)
config.py               Dataclass-based configuration (DataConfig, ModelConfig, TrainingConfig)
models.py               LocationRegressor CNN with configurable conv channels
training.py             UnifiedTrainer + LocationRegressorTrainer with TensorBoard logging
data.py                 NASA EPIC API client + async image downloader
datasets.py             PyTorch Dataset, DataLoaders, CoordinateNormalizer
eda.py                  EDA with PCA, K-Means, statistics, and overview figure
visualization.py        Plotting (distributions, world maps, training curves, error maps)
evaluation_reporter.py  Comprehensive metrics + report generation (JSON/MD/CSV)
tensorboard_utils.py    TensorBoard start/stop utilities with fallback methods
api_key_manager.py      NASA API key management
test_predictions.py     Prediction testing + world error map
__init__.py             Package init
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
- **Features**: Gradient clipping, checkpointing (best model), TensorBoard logging (scalars, graph, hyperparameters, embeddings)
- **Coordinate evaluation**: Mean/median error in degrees + Haversine distance in km

## Evaluation

`python main.py evaluate models/regressor_final.pth` computes:

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
