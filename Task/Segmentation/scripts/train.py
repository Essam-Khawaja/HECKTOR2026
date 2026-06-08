#!/usr/bin/env python3
"""Training script for HECKTOR segmentation models."""

import os
import sys
from tqdm import tqdm
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UNet3DConfig, SegResNetConfig, UNETRConfig, SwinUNETRConfig
from models import UNet3DModel, SegResNetModel, UNETRModel, SwinUNETRModel
from data import get_dataloaders
from utils.losses import get_loss_function
from utils.logging import setup_logging


## REF 1: Create a single, efficient evaluation function.
## This function calculates loss and metrics in ONE pass over the data,
## removing the major inefficiency of iterating through the validation set twice.
def evaluate_epoch(model, loader, criterion, dice_metric, device, config, use_sliding_window=False):
    """
    Run evaluation for one epoch, calculating loss and Dice metric.
    """
    model.eval()
    total_loss = 0.0
    
    # Sliding window parameters are now defined once here.
    roi_size = config.spatial_size
    sw_batch_size = 4
    
    # These MONAI transforms are now created once in main() and passed in.
    post_label = AsDiscrete(to_onehot=config.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=config.num_classes)
    
    with torch.no_grad():
        for batch in loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            
            # Use sliding window inference if specified
            if use_sliding_window:
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                    sigma_scale=0.125,
                    padding_mode="constant",
                    cval=0.0,
                    sw_device=device,
                    device=device,
                )
            else:
                outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Prepare outputs for metric calculation
            labels_list = decollate_batch(labels)
            labels_convert = [post_label(label_tensor) for label_tensor in labels_list]
            outputs_list = decollate_batch(outputs)
            outputs_convert = [post_pred(pred_tensor) for pred_tensor in outputs_list]
            
            # Update the dice metric
            dice_metric(y_pred=outputs_convert, y=labels_convert)

    avg_loss = total_loss / len(loader)
    # Aggregate the metric over all batches
    avg_dice = dice_metric.aggregate().item()
    # Reset the metric for the next epoch
    dice_metric.reset()
    
    return avg_loss, avg_dice


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(train_loader, desc='Training', leave=False):
        images, labels = batch["image"].to(device), batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    
    ## REF 2: Removed expensive Dice calculation on the training set.
    ## This speeds up training significantly. We only care about the training loss.
    return avg_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train HECKTOR segmentation model")
    parser.add_argument("--config", type=str, default="unet3d", choices=["unet3d", "segresnet", "unetr", "swinunetr"], help="Model configuration to use")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold to use (0-4)")
    parser.add_argument("--data-root", type=str, default=None, help="Path to HECKTOR training data. Defaults to HECKTOR_DATA_ROOT or ../../'HECKTOR 2026 Training Data'.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for experiment outputs")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count")
    parser.add_argument("--cache-rate", type=float, default=None, help="Override MONAI CacheDataset cache rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, help="Override device from config (e.g., 'cpu', 'cuda')")
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index")
    
    return parser.parse_args()


def validate_data_root(config, fold):
    """Fail early if the configured data root does not match expected HECKTOR files."""
    if not os.path.isdir(config.data_root):
        raise FileNotFoundError(
            f"Data root not found: {config.data_root}\n"
            "Set HECKTOR_DATA_ROOT or pass --data-root. On ARC this is likely:\n"
            "\"../../HECKTOR 2026 Training Data\" from the repository root."
        )

    if not os.path.exists(config.splits_file):
        raise FileNotFoundError(f"Splits file not found: {config.splits_file}")

    import json

    with open(config.splits_file, "r") as f:
        splits = json.load(f)

    if fold >= len(splits):
        raise ValueError(f"Fold {fold} is out of range for {config.splits_file}")

    first_case = splits[fold]["train"][0]
    case_dir = os.path.join(config.data_root, first_case)
    expected_files = [
        os.path.join(case_dir, f"{first_case}__CT.nii.gz"),
        os.path.join(case_dir, f"{first_case}__PT.nii.gz"),
        os.path.join(case_dir, f"{first_case}.nii.gz"),
    ]
    missing = [path for path in expected_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Configured data root exists, but expected HECKTOR files are missing.\n"
            f"Data root: {config.data_root}\n"
            f"Checked case: {first_case}\n"
            "Missing:\n  " + "\n  ".join(missing)
        )


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup configuration
    if args.config == "unet3d":
        # Pass fold directly to config to create proper directory structure
        config = UNet3DConfig(fold=args.fold)
    elif args.config == "segresnet":
        config = SegResNetConfig(fold=args.fold)
    elif args.config == "unetr":
        config = UNETRConfig(fold=args.fold)
    elif args.config == "swinunetr":
        config = SwinUNETRConfig(fold=args.fold)
    else:
        raise ValueError(f"Unknown config: {args.config}")

    if args.data_root:
        config.data_root = os.path.abspath(os.path.expanduser(args.data_root))
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.cache_rate is not None:
        config.cache_rate = args.cache_rate
    config.setup_output_dirs()
    
    # Override device if specified
    if args.device:
        config.device = args.device

    validate_data_root(config, args.fold)
    
    # Setup logging
    logger = setup_logging(config.log_dir)
    logger.info("Starting training...")
    logger.info(f"Configuration: {config}")
    
    # Setup device
    ## REF 4: Simplified device setup logic.
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        torch.cuda.set_device(device)
        logger.info(f"Using {device}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        if config.device == "cuda":
            logger.warning("CUDA not available, falling back to CPU.")
        logger.info(f"Using device: {device}")

    # Create model
    if args.config == "unet3d":
        model = UNet3DModel(config).to(device)
    elif args.config == "segresnet":
        model = SegResNetModel(config).to(device)
    elif args.config == "unetr":
        model = UNETRModel(config).to(device)
    elif args.config == "swinunetr":
        model = SwinUNETRModel(config).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.config}")
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup data
    train_loader, val_loader = get_dataloaders(config, fold=args.fold)
    logger.info(f"Data loaded for fold {args.fold}: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Setup training components
    criterion = get_loss_function("dice_ce")
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Create PolyLR scheduler
    scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=config.num_epochs,
        power=config.poly_lr_power,
        verbose=False
    )
    writer = SummaryWriter(config.log_dir) if config.use_tensorboard else None
    
    ## REF 5: Initialize MONAI metrics once here, preventing re-creation.
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_dice = 0.0
    if args.resume:
        checkpoint = model.load_checkpoint(args.resume, device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_dice = checkpoint.get("best_dice", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, previous best Dice: {best_val_dice:.4f}")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"Epoch {epoch}/{config.num_epochs}")
        
        # Train for one epoch and log the loss
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train - Loss: {train_loss:.4f}")
        
        # Log training loss to TensorBoard every epoch
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            
        # --- Validation ---
        ## REF 6: Simplified and cleaned up the validation and logging logic.
        val_loss, val_dice = 0.0, 0.0
        should_validate = (epoch + 1) % 5 == 0 or (epoch + 1) == config.num_epochs

        if should_validate:
            logger.info("Running validation with sliding window inference...")
            val_loss, val_dice = evaluate_epoch(
                model, val_loader, criterion, dice_metric, device, config, use_sliding_window=True
            )
            logger.info(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
            
            if writer:
                writer.add_scalar("Loss/validation", val_loss, epoch)
                writer.add_scalar("Dice/validation", val_dice, epoch)

        # PolyLR scheduler steps every epoch, regardless of validation
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Log learning rate to TensorBoard
        if writer:
            writer.add_scalar("Learning_Rate", current_lr, epoch)

        # Save best model if validation was performed and dice score improved
        if should_validate and val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
            model.save_checkpoint(best_path, epoch, optimizer.state_dict(), best_dice=best_val_dice)
            logger.info(f"New best model saved with Dice: {best_val_dice:.4f}")

        # --- Checkpointing ---
        # Save last model checkpoint periodically or at the end of training
        # Handle case when save_checkpoint_every is 0 (only save at the end)
        should_save_checkpoint = False
        if config.save_checkpoint_every > 0:
            should_save_checkpoint = (epoch + 1) % config.save_checkpoint_every == 0
        
        if should_save_checkpoint or (epoch + 1) == config.num_epochs:
            last_model_path = os.path.join(config.checkpoint_dir, "last_model.pth")
            model.save_checkpoint(last_model_path, epoch, optimizer.state_dict(), best_dice=best_val_dice)
            logger.info(f"Saved last model checkpoint at epoch {epoch}")

    logger.info("Training completed!")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
