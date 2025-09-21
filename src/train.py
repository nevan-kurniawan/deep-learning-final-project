import argparse
import pathlib

import timm
import yaml
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import os

from src.model import ModifiedModel
from src.data_loader import create_dataloaders
from src.utils import set_seed

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG = ROOT_DIR / 'config' / 'config.yaml'

def train(config: dict):
    print("Starting training...")

    set_seed(config['train_params']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = ROOT_DIR / config['model_params']['output_dir']
    output_dir.mkdir(parents=False, exist_ok=True)

    train_loader, val_loader, _, class_names = create_dataloaders(
        data_dir=ROOT_DIR / config['data_params']['data_dir'],
        batch_size=config['data_params']['batch_size'],
        image_size=config['data_params']['image_size']
    )
    print(f"Data loaded. Found {len(class_names)} classes.")

    print("Instantiating model...")

    base_model = timm.create_model('deit_tiny_patch16_224', pretrained=True)

    model = ModifiedModel(
        base_model = base_model,
        num_classes = config['model_params']['num_classes'],
        lora_rank = config['model_params']['lora_rank'],
        lora_alpha= config['model_params']['lora_alpha'],
        num_experts= config['model_params']['num_experts'],
        k = config['model_params']['k'],
    ).to(device)

    # Filter for trainable parameters (LoRA experts and the new classifier head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config['train_params']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    aux_loss_weight = config['train_params']['aux_loss_weight']

    best_val_accuracy = 0.0
    print("Setup complete. Starting training loop...")