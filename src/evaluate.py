import argparse
import pathlib
import yaml
import torch
import timm
import pandas as pd
from tqdm import tqdm

# Import your custom modules
from src.model import ModifiedModel
from src.data_loader import create_dataloaders

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def evaluate(model_path: pathlib.Path, log_dir: pathlib.Path):
    """
    Runs inference on the test set.

    Args:
        model_path (pathlib.Path): Path to the saved model state dictionary.
        log_dir (pathlib.Path): Directory to save the prediction logs.
    """
    print("--- Starting evaluation inference run ---")

    # --- 1. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    log_dir.mkdir(parents=True, exist_ok=True) # Use the passed argument

    # --- 2. DATA LOADING ---
    checkpoint = torch.load(model_path, map_location=device)
    data_config = checkpoint['data_config']
    class_names = checkpoint['class_names']

    # Now the create_dataloaders call only needs to return the loader
    test_loader, = create_dataloaders(
        data_dir=PROJECT_ROOT / data_config['data_dir'],
        batch_size=data_config['batch_size'],
        image_size=data_config['image_size'],
        num_workers=data_config.get('num_workers', 4),
        splits=['test']
    )
    print(f"Test data loaded. Found {len(class_names)} classes.")

    # --- 3 & 4. LOAD MODEL FROM SELF-CONTAINED CHECKPOINT ---
    # Re-create the model architecture using the *saved* configuration from the checkpoint
    base_model = timm.create_model(
        checkpoint['model_config']['base_model_name'], # This can also be parameterized
        pretrained=False
    )
    model_config = checkpoint['model_config']
    # Overwrite num_classes with the actual count from the dataset
    model_config['num_classes'] = len(class_names)

    model = ModifiedModel(
        base_model=base_model,
        **model_config
    )

    # Load the saved weights into the correctly structured model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # --- 5. INFERENCE LOOP ---
    all_filepaths = []
    all_true_labels_idx = []
    all_top5_preds_idx = []

    with torch.no_grad():
        # Unpack all three items directly from the DataLoader
        for images, labels, filepaths in tqdm(test_loader, desc="Generating Predictions"):
            images = images.to(device)

            logits, _ = model(images)
            _, top5_indices = torch.topk(logits, k=5, dim=1)

            # The association is now guaranteed to be correct
            all_filepaths.extend(filepaths)
            all_true_labels_idx.extend(labels.numpy())
            all_top5_preds_idx.extend(top5_indices.cpu().numpy())

    # --- 6. SAVE PREDICTIONS TO FILE ---
    # The predicted_top5_idx column will contain lists of 5 integers
    predictions_df = pd.DataFrame({
        'filepath': all_filepaths,
        'true_label_idx': all_true_labels_idx,
        'predicted_top5_idx': [list(preds) for preds in all_top5_preds_idx]
    })

    save_path = log_dir / "test_predictions.csv"
    predictions_df.to_csv(save_path, index=False)

    print("\n--- Inference Complete ---")
    print(f"Raw predictions saved to {save_path}")


# evaluate.py (Modified __main__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for MoE-LoRA DeiT model.")
    # REMOVED the config argument
    # parser.add_argument(
    #     "-c", "--config", type=str, required=True,
    #     help="Path to the YAML configuration file used for training."
    # )
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, # Make it required
        help="Path to the saved .pth model file."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="outputs/logs",
        help="Directory to save the output predictions CSV."
    )
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model_path
    log_dir = PROJECT_ROOT / args.output_dir

    # REMOVED the config loading logic
    # with open(config_filepath, 'r') as f:
    #     config = yaml.safe_load(f)

    evaluate(model_path, log_dir)