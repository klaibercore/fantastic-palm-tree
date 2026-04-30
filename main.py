"""
Simplified main script with consolidated commands and improved structure.
"""

import logging
import argparse
import random
import torch
import numpy as np
from pathlib import Path

from config import Config
from data import EPICDataDownloader, CoordinateExtractor
from datasets import create_dataloaders
from models import create_location_regressor
from training import LocationRegressorTrainer
from visualization import (
    plot_coordinate_distribution,
    plot_world_map_with_coordinates,
    create_coordinate_statistics_table
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across PyTorch, NumPy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def setup_data_pipeline(config):
    """Setup the complete data pipeline."""
    logger.info("Setting up data pipeline...")
    
    # Initialize downloader
    downloader = EPICDataDownloader(config)
    
    # Download metadata if needed
    if not Path(config.data.raw_data_dir).exists():
        logger.info("Downloading metadata...")
        downloader.download_metadata()
    
    # Download daily data if needed
    if not Path(config.data.images_dir).exists():
        logger.info("Downloading daily data...")
        downloader.download_recent(7)
    
    logger.info("Data pipeline setup complete!")
    
    # Create coordinate statistics
    extractor = CoordinateExtractor(config)
    lat_coords, lon_coords = extractor.extract_coordinates()
    
    if lat_coords and lon_coords:
        logger.info(f"Coordinate range: Lat [{min(lat_coords):.3f}, {max(lat_coords):.3f}], "
                    f"Lon [{min(lon_coords):.3f}, {max(lon_coords):.3f}]")
        
        # Create visualizations
        logger.info("Creating coordinate distribution plots...")
        plot_coordinate_distribution(lat_coords, lon_coords, save_path="outputs/coordinate_distribution.png", show_plot=False)
        plot_world_map_with_coordinates(lat_coords, lon_coords, save_path="outputs/coordinate_world_map.png", show_plot=False)
        stats_table = create_coordinate_statistics_table(lat_coords, lon_coords)
        
        # Save statistics table
        import pandas as pd
        stats_table.to_csv("outputs/coordinate_statistics.csv", index=False)
        logger.info(f"Coordinate statistics saved to outputs/coordinate_statistics.csv")
    else:
        logger.warning("No coordinates found for visualization")


def train_model(config, model_type: str = "regressor"):
    """Unified training function for both model types."""
    logger.info(f"Training {model_type} model...")
    
    # Create model and data
    if model_type != "regressor":
        raise ValueError(f"Unknown model type: {model_type}")
    model = create_location_regressor(config)
    trainer_class = LocationRegressorTrainer
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # Setup directories
    import os
    os.makedirs(config.training.log_dir, exist_ok=True)
    os.makedirs(config.training.save_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Create timestamped run directory info
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{config.training.log_dir}/run_{timestamp}"
    logger.info(f"TensorBoard logs will be saved to: {run_dir}")
    logger.info(f"Models will be saved to: {config.training.save_dir}")
    logger.info(f"To view TensorBoard: tensorboard --logdir {config.training.log_dir}")
    
    # Create trainer
    trainer = trainer_class(model, train_loader, val_loader, config)
    
    try:
        # Train model
        results = trainer.train()
        logger.info(f"Training complete! Best validation loss: {results['best_val_loss']:.4f}")
        
        # Save final model as checkpoint
        final_model_path = os.path.join(config.training.save_dir, f"{model_type}_final.pth")
        torch.save({'model_state_dict': model.state_dict()}, final_model_path)
        logger.info(f"Model saved to: {final_model_path}")
        
        return final_model_path
        
    finally:
        # Cleanup
        trainer.cleanup()


def evaluate_model_performance(config, model_path: str):
    """Unified model evaluation with comprehensive metrics."""
    logger.info(f"Evaluating model: {model_path}")
    
    # Load model and create data
    model = create_location_regressor(config)
    
    # Handle PyTorch 2.6+ weights_only security
    try:
        # Try loading as checkpoint first (most common case)
        checkpoint = torch.load(model_path, map_location=config.training.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model checkpoint successfully")
        else:
            # Checkpoint doesn't have model_state_dict, treat entire checkpoint as weights
            model.load_state_dict(checkpoint)
            logger.info("Loaded model weights successfully")
    except Exception as e:
        if "weights_only" in str(e):
            # Fallback for PyTorch 2.6+ security restrictions
            try:
                torch.serialization.add_safe_globals([Config])
                checkpoint = torch.load(model_path, map_location=config.training.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info("Loaded model checkpoint successfully (safe globals fallback)")
            except Exception as fallback_e:
                raise Exception(f"Failed to load model: {e}, {fallback_e}")
        else:
            # Original exception wasn't weights_only related, re-raise it
            raise e
    
    _, _, test_loader = create_dataloaders(config)
    
    # Create trainer for evaluation with dummy train loader
    from torch.utils.data import DataLoader, TensorDataset
    dummy_train_data = TensorDataset(torch.zeros(1, 1, 64, 64), torch.zeros(1, 2))
    dummy_train_loader = DataLoader(dummy_train_data, batch_size=1)
    
    trainer = LocationRegressorTrainer(model, dummy_train_loader, test_loader, config)
    
    # Evaluate with coordinate metrics
    metrics = trainer.evaluate_coordinates(test_loader)
    
    logger.info("Evaluation Results:")
    logger.info(f"  Mean coordinate error: {metrics['mean_coordinate_error_deg']:.4f}°")
    logger.info(f"  Median coordinate error: {metrics['median_coordinate_error_deg']:.4f}°")
    logger.info(f"  Mean Haversine distance: {metrics['mean_haversine_km']:.1f} km")
    logger.info(f"  Median Haversine distance: {metrics['median_haversine_km']:.1f} km")
    
    return metrics


def download_data(config, num_days: int = 7):
    """Download recent satellite images from the last N days."""
    logger.info(f"Downloading images from last {num_days} days...")
    
    downloader = EPICDataDownloader(config)
    downloader.download_recent(num_days)
    
    logger.info("Download complete!")


def main():
    """Main function with simplified command structure."""
    parser = argparse.ArgumentParser(
        description="Satellite Image Coordinate Prediction - Consolidated Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
  %(prog)s setup                    # Setup complete data pipeline
  %(prog)s train regressor          # Train location regressor
  %(prog)s evaluate model.pth       # Evaluate trained model
  %(prog)s download 7               # Download last 7 days
  %(prog)s download 30              # Download last 30 days
        """
    )
    
    # Main command
    parser.add_argument("command", 
                       choices=["setup", "train", "evaluate", "download"],
                       help="Command to execute")
    
    # Command-specific arguments
    parser.add_argument("target", nargs="?",
                       help="Target for command (model type, model path, or number of days)")
    parser.add_argument("value", nargs="?", type=int,
                       help="Deprecated: use target instead")
    
    # Configuration and overrides
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "cpu"],
                       help="Training device (auto-detects: cuda > mps > cpu)")
    parser.add_argument("--no-tensorboard", action="store_true",
                       help="Disable automatic TensorBoard launch")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config.from_dict(config_dict)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.epochs:
        config.training.max_epochs = args.epochs
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.training.device = args.device
    if args.no_tensorboard:
        config.training.launch_tensorboard = False
    
    # Set threading
    torch.set_num_interop_threads(config.training.num_threads)
    torch.set_num_threads(config.training.num_threads)

    # Set seeds for reproducibility
    _set_seed(config.training.random_seed)
    
    try:
        # Execute command
        if args.command == "setup":
            setup_data_pipeline(config)
            
        elif args.command == "train":
            if not args.target or args.target != "regressor":
                raise ValueError("Training requires specifying 'regressor'")
            train_model(config, args.target)
            
        elif args.command == "evaluate":
            if not args.target:
                raise ValueError("Evaluation requires specifying model path")
            evaluate_model_performance(config, args.target)
            
        elif args.command == "download":
            num_days = int(args.target) if args.target else 7
            download_data(config, num_days)
            
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())