"""
Training utilities for satellite image coordinate prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import os
import logging
import time
from typing import Optional, Dict
from tqdm import tqdm
from datetime import datetime

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None

from torch.utils.data import DataLoader

from models import LocationRegressor
from datasets import CoordinateNormalizer
from tensorboard_utils import is_port_available, start_tensorboard

logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """Trainer with TensorBoard logging of relevant metrics only."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device(config.training.device)

        self.model.to(self.device)

        self.optimizer = self._setup_optimizer()
        self.criterion = self._setup_criterion()
        self.scheduler = self._setup_scheduler()
        self._setup_coordinate_normalizer()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # TensorBoard
        self.tensorboard_process = None
        self.writer = None
        if SummaryWriter is not None:
            self._setup_tensorboard()

    # ── Setup ──

    def _setup_coordinate_normalizer(self):
        all_coords = []
        for _, coords in self.train_loader:
            all_coords.append(coords)
        if all_coords:
            self.coord_normalizer = CoordinateNormalizer(torch.cat(all_coords, dim=0))
        else:
            self.coord_normalizer = CoordinateNormalizer()

    def _setup_optimizer(self):
        name = self.config.training.optimizer.lower()
        params = self.model.parameters()
        lr = self.config.training.learning_rate
        wd = self.config.training.weight_decay
        if name == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=wd)
        elif name == 'sgd':
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        elif name == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {name}")

    def _setup_criterion(self):
        name = self.config.training.loss_function.lower()
        if name == 'mse':
            return nn.MSELoss()
        elif name == 'l1':
            return nn.L1Loss()
        elif name == 'smooth_l1':
            return nn.SmoothL1Loss()
        raise ValueError(f"Unknown loss function: {name}")

    def _setup_scheduler(self):
        name = self.config.training.scheduler.lower()
        if name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma,
            )
        elif name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.max_epochs
            )
        elif name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.training.gamma,
                patience=self.config.training.step_size,
            )
        elif name == 'none':
            return None
        raise ValueError(f"Unknown scheduler: {name}")

    # ── TensorBoard ──

    def _setup_tensorboard(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.tensorboard_run_dir = os.path.join(
                self.config.training.log_dir, f"run_{timestamp}"
            )
            os.makedirs(self.tensorboard_run_dir, exist_ok=True)

            self.writer = SummaryWriter(log_dir=self.tensorboard_run_dir)

            # Log model architecture once at init; hparams logged after training
            self.writer.add_text('model/architecture', f"```\n{self.model}\n```")

            # Log computational graph
            try:
                dummy_input = torch.zeros(
                    1, self.config.model.input_channels,
                    self.config.data.image_size, self.config.data.image_size,
                ).to(self.device)
                self.writer.add_graph(self.model, dummy_input)
                logger.info("Model graph logged to TensorBoard")
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")

            if self.config.training.launch_tensorboard:
                self._launch_tensorboard()

            logger.info(f"TensorBoard logs: {self.tensorboard_run_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize TensorBoard: {e}")
            self.writer = None

    def _log_hparams(self, metrics: Dict[str, float]):
        """Log hyperparameters with final training metrics to TensorBoard."""
        if self.writer is None:
            return
        hparams = {
            'lr': self.config.training.learning_rate,
            'batch_size': self.config.training.batch_size,
            'epochs': self.config.training.max_epochs,
            'optimizer': self.config.training.optimizer,
            'loss_fn': self.config.training.loss_function,
            'scheduler': self.config.training.scheduler,
            'weight_decay': self.config.training.weight_decay,
            'device': str(self.device),
            'image_size': self.config.data.image_size,
            'conv_channels': str(self.config.model.conv_channels),
            'hidden_dim': self.config.model.hidden_dim,
            'params': sum(p.numel() for p in self.model.parameters()),
        }
        self.writer.add_hparams(
            hparam_dict=hparams,
            metric_dict=metrics,
            run_name='.',
        )

    def _log_embeddings(self):
        """Log model embeddings from the validation set to TensorBoard."""
        if self.writer is None:
            return

        self.model.eval()
        all_embeddings = []
        all_labels = []
        all_images = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                embeddings = self.model.get_embeddings(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(targets)
                all_images.append(images.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_images = torch.cat(all_images, dim=0)

        # Build metadata: lon, lat per sample
        metadata = [
            [f"{lon:.1f}", f"{lat:.1f}"]
            for lon, lat in all_labels.tolist()
        ]
        metadata_header = ["longitude", "latitude"]

        # Normalize images to [0,1] for thumbnails; expand grayscale to 3-ch
        label_img = all_images
        if label_img.shape[1] == 1:
            label_img = label_img.repeat(1, 3, 1, 1)

        self.writer.add_embedding(
            all_embeddings,
            metadata=metadata,
            metadata_header=metadata_header,
            label_img=label_img,
            global_step=self.current_epoch,
            tag="model_embeddings",
        )
        logger.info(f"Logged {len(all_embeddings)} embeddings to TensorBoard")

    def _launch_tensorboard(self):
        try:
            success = start_tensorboard(
                self.tensorboard_run_dir, 6006, open_browser=False
            )
            if success:
                self.tensorboard_process = True
                logger.info("TensorBoard launched at http://localhost:6006")
            else:
                self.tensorboard_process = None
        except Exception as e:
            logger.error(f"Failed to launch TensorBoard: {e}")
            self.tensorboard_process = None

    # ── Training loop ──

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, targets in tqdm(self.train_loader, desc='Training', leave=False):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(
                self.coord_normalizer.normalize(outputs),
                self.coord_normalizer.normalize(targets)
            )
            loss.backward()

            if self.config.training.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clipping
                )

            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        if self.writer is not None:
            self.writer.add_scalar('loss/train', avg_loss, self.current_epoch)

        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation', leave=False):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(
                    self.coord_normalizer.normalize(outputs),
                    self.coord_normalizer.normalize(targets)
                )
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        if self.writer is not None:
            self.writer.add_scalar('loss/val', avg_loss, self.current_epoch)

        return avg_loss

    def train(self):
        logger.info(f"Training for {self.config.training.max_epochs} epochs on {self.device}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in tqdm(range(self.config.training.max_epochs), desc="Epochs"):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log learning rate
            if self.writer is not None:
                self.writer.add_scalar(
                    'lr', self.optimizer.param_groups[0]['lr'], epoch
                )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')

            logger.info(
                f'Epoch {epoch+1}/{self.config.training.max_epochs}: '
                f'Train={train_loss:.4f}  Val={val_loss:.4f}'
            )

        logger.info('Training complete!')

        # Log model embeddings for TensorBoard projector
        self._log_embeddings()

        # Log hparams with final metrics so TensorBoard can compare runs
        self._log_hparams({
            'hparam/best_val_loss': self.best_val_loss,
            'hparam/final_train_loss': self.train_losses[-1],
            'hparam/final_val_loss': self.val_losses[-1],
        })

        if self.writer is not None:
            self.writer.flush()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }

    # ── Checkpoint ──

    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
        }
        filepath = os.path.join(self.config.training.save_dir, filename)
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f'Checkpoint saved: {filepath}')

    def load_checkpoint(self, filename: str):
        filepath = os.path.join(self.config.training.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f'Checkpoint loaded: {filepath}')

    def cleanup(self):
        if self.writer is not None:
            self.writer.close()
        if isinstance(self.tensorboard_process, subprocess.Popen):
            try:
                self.tensorboard_process.terminate()
                self.tensorboard_process.wait(timeout=5)
            except Exception:
                try:
                    self.tensorboard_process.kill()
                except Exception:
                    pass
        self.tensorboard_process = None


class LocationRegressorTrainer(UnifiedTrainer):
    """Trainer for location regression with coordinate evaluation."""

    def __init__(self, model: LocationRegressor, train_loader: DataLoader,
                 val_loader: DataLoader, config, device: Optional[str] = None):
        super().__init__(model, train_loader, val_loader, config, device)

    def evaluate_coordinates(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate coordinate prediction accuracy."""
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                predictions = self.model(images)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            coord_errors = self.coord_normalizer.compute_coordinate_error_degrees(
                all_predictions, all_targets
            )
            haversine_distances = self.coord_normalizer.compute_haversine_distance(
                all_predictions, all_targets
            )

            return {
                'mean_coordinate_error_deg': coord_errors.mean().item(),
                'median_coordinate_error_deg': coord_errors.median().item(),
                'mean_haversine_km': haversine_distances.mean().item(),
                'median_haversine_km': haversine_distances.median().item(),
            }
