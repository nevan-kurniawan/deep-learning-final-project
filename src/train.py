"""
Main training script for the vision model.

This script handles the entire training and validation pipeline, including:
- Loading configuration files (handling inheritance)
- Setting up logging and seeding
- Creating data loaders
- Initializing the model (including PEFT modifications)
- Running the training and validation loops
- Saving checkpoints and training history.
"""

import argparse
import json
import pathlib
import torch
import timm
import logging
import pandas as pd
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from timm.data import Mixup

# Import custom modules
from src.model import ModifiedModel
from src.utils import set_seed, get_parameter_summary, setup_logging, get_gflops, load_config
from src.data_loader import create_dataloaders

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def train(config: dict, experiment_name: str):
    """
    Main script for training the vision model based on a provided config file.

    Args:
        config: The dictionary containing the configuration for the experiment (Loaded from the YAML config file).
        experiment_name: The name of the experiment (e.g. 'deit_moe_lora', 'vit_full_finetune')
    """

    ### 1. Setup
    logging.info(f"Starting experiment: {experiment_name}")

    output_dir = PROJECT_ROOT / config['output_params']['model_save_dir'] / experiment_name
    log_dir = PROJECT_ROOT / config['output_params']['log_dir'] / experiment_name

    log_filepath = setup_logging(log_dir)
    logging.info(f"Configuration used: {config}")

    set_seed(config['train_params']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    ### 2. Data loading
    dataloaders, class_names = create_dataloaders(
        data_dir=PROJECT_ROOT / config['data_params']['data_dir'],
        batch_size=config['data_params']['batch_size'],
        image_size=config['data_params']['image_size'],
        splits=['train', 'val'],
        num_workers=config['data_params'].get('num_workers', 4)
    )

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    logging.info(f"Data loaded. Found {len(class_names)} classes.")

    logging.info("Instantiating model...")
    base_model = timm.create_model(
        config['model_params']['base_model_name'],
        pretrained=True
    )

    model = ModifiedModel(
        base_model=base_model,
        num_classes=len(class_names),
        model_config=config['model_params']
    ).to(device)

    ### 3. Log the model summary.
    param_summary = get_parameter_summary(model)
    logging.info(f"Model Parameter Summary: {param_summary}")

    dummy_input_for_flops = torch.randn(1, 3, 224, 224).to(device)
    gflops = get_gflops(model, dummy_input_for_flops)
    logging.info(f"Total GFLOPs: {gflops:.2f}")

    logging.info("--- Model Parameter Summary ---")
    logging.info(f"Total parameters:      {param_summary['total_params']:,}")
    logging.info(f"Trainable parameters:  {param_summary['trainable_params']:,}")
    logging.info(f"Trainable %:           {param_summary['trainable_percentage']:.2f}%")
    logging.info(f"Total GFLOPs:          {gflops:.2f}")
    logging.info("-----------------------------")

    summary_path = pathlib.Path(log_filepath).parent / f"{pathlib.Path(log_filepath).stem}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(param_summary, f, indent=4)
    logging.info(f"Parameter summary saved to {summary_path}")

    ### 4. Optimizer and loss function.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config['train_params']['learning_rate'], weight_decay=config['train_params'].get('weight_decay', 0.0))

    # Setup Mixup and CutMix augmentations
    mixup_params = config['train_params'].get('mixup', {})
    mixup_fn = None
    if mixup_params and mixup_params.get('mixup_alpha', 0.0) > 0.0 or mixup_params.get('cutmix_alpha', 0.0) > 0.0:
        logging.info("Mixup/CutMix enabled.")
        mixup_fn = Mixup(
            mixup_alpha=mixup_params['mixup_alpha'],
            cutmix_alpha=mixup_params['cutmix_alpha'],
            prob=mixup_params.get('prob', 1.0),
            switch_prob=mixup_params.get('switch_prob', 0.5),
            mode=mixup_params.get('mode', 'batch'),
            label_smoothing=config['train_params'].get('label_smoothing', 0.1),
            num_classes=len(class_names)
        )

    # Setup label smoothing
    label_smoothing = config['train_params'].get('label_smoothing', 0.0)

    # Setup loss
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    logging.info(f"Using CrossEntropyLoss with label smoothing: {label_smoothing}")

    aux_loss_weight = config['train_params'].get('aux_loss_weight', 0)

    warmup_epochs = config['train_params'].get('warmup_epochs', 0)
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['train_params']['epochs'] - warmup_epochs,
        eta_min=config['train_params']['scheduler']['eta_min']
    )

    # Warmup scheduler (Linear)
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6,  # Start LR near zero
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = main_scheduler

    patience = config['train_params']['early_stopping_patience']
    early_stopping_counter = 0
    best_val_loss = float('inf')

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_aux_loss': []}
    logging.info("Setup complete. Starting training loop...")

    ### 5. Training loop
    # Setup gradient clipping
    grad_clip_norm = config['train_params'].get('grad_clip_norm')
    if grad_clip_norm:
        logging.info(f"Gradient clipping enabled with max_norm: {grad_clip_norm}")

    for epoch in range(config['train_params']['epochs']):
        # Train phase
        model.train()
        train_loss, train_correct = 0, 0
        train_aux_loss_accum = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['train_params']['epochs']} [Train]"):
            images, labels = images.to(device), labels.to(device)

            if mixup_fn:
                images, labels = mixup_fn(images, labels)

            optimizer.zero_grad()
            logits, total_aux_loss = model(images)
            main_loss = criterion(logits, labels)
            total_loss = main_loss + aux_loss_weight * total_aux_loss

            total_loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            optimizer.step()

            train_loss += total_loss.item()
            train_aux_loss_accum += total_aux_loss.item()
            true_labels = labels.argmax(dim=1) if labels.ndim > 1 else labels
            train_correct += (logits.argmax(dim=1) == true_labels).sum().item()

        # Val phase
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{config['train_params']['epochs']} [Val]"):
                images, labels = images.to(device), labels.to(device)

                logits, _ = model(images)
                main_loss = criterion(logits, labels)

                val_loss += main_loss.item()
                true_labels = labels.argmax(dim=1) if labels.ndim > 1 else labels
                val_correct += (logits.argmax(dim=1) == true_labels).sum().item()

        ### 6. Logging and checkpointing
        avg_train_loss = train_loss / len(train_loader)
        avg_train_aux_loss = train_aux_loss_accum / len(train_loader)
        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / len(val_loader.dataset)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['train_aux_loss'].append(avg_train_aux_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Aux Loss: {avg_train_aux_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%| "
            f"LR: {current_lr:.1E}")

        scheduler.step()

        # Chooses and saves best model based on val_loss.
        if avg_val_loss < best_val_loss:
            previous_best = best_val_loss
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            save_path = output_dir / f"best_{experiment_name}_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'model_config': config['model_params'],
                'data_config': config['data_params'],
                'class_names': class_names
            }, save_path)
            logging.info(f"Validation loss decreased from {previous_best:.4f} to {best_val_loss:.4f}. Saving new best model.")
        else:
            early_stopping_counter += 1
            logging.info(
                f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                logging.info(f"--- Early stopping triggered after {epoch + 1} epochs. ---")
                break

    logging.info("--- Training finished ---")

    ### 7. Save training history.
    history_df = pd.DataFrame(history)
    history_path = pathlib.Path(log_filepath).parent / f"{pathlib.Path(log_filepath).stem}_history.csv"
    history_df.to_csv(history_path, index=False)
    logging.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    """
    Main entry point for the script. Parses command-line arguments, loads the specified config file,
    and initiates the training process.
    """
    parser = argparse.ArgumentParser(description="Train a vision transformer model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    config_full_path = pathlib.Path(args.config)

    config = load_config(config_full_path)

    experiment_name = config_full_path.stem

    train(config, experiment_name)