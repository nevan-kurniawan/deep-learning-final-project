import argparse
import json
import pathlib
import yaml
import torch
import timm
import logging
import pandas as pd
from datetime import datetime
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

# Import custom modules
from src.model import ModifiedModel
from src.utils import set_seed
from src.data_loader import create_dataloaders

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def get_parameter_summary(model: nn.Module) -> dict:
    """Calculates the total and trainable parameters of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params,
    }

def setup_logging(log_dir: pathlib.Path):
    """Configures the logging for the training run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = log_dir / f"training_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return log_filepath


def train(config: dict):
    """Main training and validation function."""

    # --- 1. SETUP ---
    log_dir = PROJECT_ROOT / config['output_params']['log_dir']
    log_filepath = setup_logging(log_dir)
    logging.info("--- Starting training run ---")
    logging.info(f"Configuration used: {config}")

    set_seed(config['train_params']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    output_dir = PROJECT_ROOT / config['output_params']['model_save_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. DATA LOADING ---
    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=PROJECT_ROOT / config['data_params']['data_dir'],
        batch_size=config['data_params']['batch_size'],
        image_size=config['data_params']['image_size'],
        splits = ['train', 'val'],
        num_workers=config['data_params']['num_workers']
    )
    logging.info(f"Data loaded. Found {len(class_names)} classes.")

    # --- 3. MODEL INSTANTIATION ---
    logging.info("Instantiating model...")
    base_model = timm.create_model(
        config['model_params']['base_model_name'],
        pretrained=True
    )

    model = ModifiedModel(
        base_model=base_model,
        target_blocks_indices=config['model_params']['target_blocks_indices'],
        num_classes=len(class_names),
        lora_rank=config['model_params']['lora_rank'],
        lora_alpha=config['model_params']['lora_alpha'],
        num_experts=config['model_params']['num_experts'],
        k=config['model_params']['k']
    ).to(device)

    # Param count
    param_summary = get_parameter_summary(model)

    # Log the summary to the console/log file
    logging.info("--- Model Parameter Summary ---")
    logging.info(f"Total parameters:      {param_summary['total_params']:,}")
    logging.info(f"Trainable parameters:  {param_summary['trainable_params']:,}")
    logging.info(f"Trainable %:           {param_summary['trainable_percentage']:.2f}%")
    logging.info("-----------------------------")

    # Save the summary to a JSON file
    summary_path = pathlib.Path(log_filepath).parent / f"{pathlib.Path(log_filepath).stem}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(param_summary, f, indent=4)
    logging.info(f"Parameter summary saved to {summary_path}")

    # --- 4. OPTIMIZER AND LOSS FUNCTION ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config['train_params']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    aux_loss_weight = config['train_params']['aux_loss_weight']

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['train_params']['scheduler']['T_max'],
        eta_min=config['train_params']['scheduler']['eta_min']
    )

    patience = config['train_params']['early_stopping_patience']
    early_stopping_counter = 0
    best_val_loss = float('inf')

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    logging.info("Setup complete. Starting training loop...")

    # --- 5. TRAINING LOOP ---
    for epoch in range(config['train_params']['epochs']):
        # -- Training Phase --
        model.train()
        train_loss, train_correct = 0, 0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['train_params']['epochs']} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, total_aux_loss = model(images)
            main_loss = criterion(logits, labels)
            total_loss = main_loss + aux_loss_weight * total_aux_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()

        # -- Validation Phase --
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader,
                                          desc=f"Epoch {epoch + 1}/{config['train_params']['epochs']} [Val]"):
                images, labels = images.to(device), labels.to(device)

                logits, _ = model(images)
                main_loss = criterion(logits, labels)

                val_loss += main_loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()

        # --- 6. LOGGING AND CHECKPOINTING ---
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / len(val_loader.dataset)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        logging.info(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                     f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            previous_best = best_val_loss
            best_val_loss = avg_val_loss
            early_stopping_counter = 0 # Reset counter
            save_path = output_dir / "best_model.pth"
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

    # --- 7. SAVE TRAINING HISTORY ---
    history_df = pd.DataFrame(history)
    history_path = pathlib.Path(log_filepath).parent / f"{pathlib.Path(log_filepath).stem}_history.csv"
    history_df.to_csv(history_path, index=False)
    logging.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for MoE-LoRA DeiT model.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Name of the configuration file in the 'config/' directory."
    )
    args = parser.parse_args()

    config_filepath = CONFIG_DIR / args.config
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)

    train(config)