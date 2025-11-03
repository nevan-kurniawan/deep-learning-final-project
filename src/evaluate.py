"""
Main script for running evaluation on a trained model checkpoint.

This script loads a model, runs it on the specified test set,
and saves the raw model predictions to a CSV file.

It can optionally run diagnostics on MoE-LoRA models to capture
and save router outputs for further analysis.
"""

import argparse
import pathlib
import torch
import timm
import pandas as pd
from tqdm import tqdm
import logging
import numpy as np
from collections import defaultdict
from functools import partial

# Import modules
from src.model import ModifiedModel, MoE_LoRA
from src.data_loader import create_dataloaders
from src.utils import setup_logging

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def evaluate(model_path: pathlib.Path, log_dir: pathlib.Path, run_diagnostics: bool = False):
    """
    Runs the inference on the test set using a trained model checkpoint. This function
    saves the top-5 predicted indices and true labels for each sample into a .csv file.
    Optionally runs router diagnostics for MoE-LoRA models by saving the last routing layer's routing indices and weights.
    Args:
        model_path: path to the saved model.
        log_dir: path to the directory where the log file is to be written.
        run_diagnostics: Argument to run MoE-LoRA diagnostics or not. Defaults to False.
            The diagnostics is done by saving the outputs (indices and weights) for MoE router analysis.
    """
    logging.info("--- Starting evaluation inference run ---")

    all_router_outputs = defaultdict(lambda: {'weights': [], 'indices': []})

    def routing_hook(module, input, output, layer_name: str):
        """
        Appends router outputs (weights and indices) to a dictionary keyed by layer_name.
        Args:
            module: Unused module argument.
            input: Unused input argument.
            output: The router output tuple: (top_k_weights, top_k_indices, aux_loss)
            layer_name (str): The name of the layer this hook is attached to.
        """
        all_router_outputs[layer_name]['weights'].append(output[0].cpu())
        all_router_outputs[layer_name]['indices'].append(output[1].cpu())

    ### 1. Setup
    log_filepath = setup_logging(log_dir)
    logging.info(f"Using log file: {log_filepath}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if run_diagnostics:
        logging.info("Routing diagnostics requested.")

    ### 2. Data loading
    checkpoint = torch.load(model_path, map_location=device)
    data_config = checkpoint['data_config']
    class_names = checkpoint['class_names']

    dataloaders, _ = create_dataloaders(
        data_dir=PROJECT_ROOT / data_config['data_dir'],
        batch_size=data_config['batch_size'],
        image_size=data_config['image_size'],
        num_workers=data_config.get('num_workers', 4),
        splits=['test']
    )
    test_loader = dataloaders['test']
    logging.info(f"Test data loaded. Found {len(class_names)} classes.")

    ### 3. Load model from checkpoint
    base_model = timm.create_model(
        checkpoint['model_config']['base_model_name'],
        pretrained=False
    )
    model_config = checkpoint['model_config']

    model = ModifiedModel(
        base_model=base_model,
        num_classes=len(class_names),
        model_config=model_config
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    ### 4. Register routing hook to routing layers if requested.
    hook_handles = []
    if run_diagnostics:
        logging.info("Registering routing hooks on MoE_LoRA layers...")
        for layer_name, module in model.named_modules():
            if isinstance(module, MoE_LoRA):
                if hasattr(module, 'router') and isinstance(module.router, torch.nn.Module):
                    # functools.partial is used to create a function that can preserve the module, input, output contract
                    # for .register_forward_hook by attaching the layer_name parameter directly
                    hook_with_name = partial(routing_hook, layer_name=layer_name)
                    handle = module.router.register_forward_hook(hook_with_name)
                    hook_handles.append(handle)
                    logging.info(f"Hook registered on: {layer_name}")
        if not hook_handles:
            logging.warning("Run diagnostics requested, but no MoE_LoRA layers found.")

    ### 5. Inference loop
    all_indices = []
    all_true_labels_idx = []
    all_top5_preds_idx = []
    current_idx = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Generating Predictions"):
            images = images.to(device)

            logits, _ = model(images)
            _, top5_indices = torch.topk(logits, k=5, dim=1)

            batch_size = images.shape[0]
            indices_in_batch = list(range(current_idx, current_idx + batch_size))
            current_idx += batch_size

            all_indices.extend(indices_in_batch)
            all_true_labels_idx.extend(labels.numpy())
            all_top5_preds_idx.extend(top5_indices.cpu().numpy())

        if hook_handles:
            logging.info("Removing routing hooks...")
            for handle in hook_handles:
                handle.remove()

        ### 6. Save predictions to file
        logging.info("Creating predictions DataFrame...")
        predictions_df = pd.DataFrame({
            'image_index': all_indices,
            'true_label_idx': all_true_labels_idx,
            'predicted_top5_idx': [arr.tolist() for arr in all_top5_preds_idx]
        })
        save_path = log_dir / "test_predictions.csv"
        predictions_df.to_csv(save_path, index=False)
        logging.info(f"Raw predictions saved to {save_path}")

        if run_diagnostics and all_router_outputs:
            try:
                logging.info("Processing and saving routing diagnostics...")
                npz_data = {}
                for layer_name, outputs in all_router_outputs.items():
                    key_weights = f"{layer_name}_weights"
                    key_indices = f"{layer_name}_indices"

                    npz_data[key_weights] = torch.cat(outputs['weights'], dim=0).numpy()
                    npz_data[key_indices] = torch.cat(outputs['indices'], dim=0).numpy()
                    logging.info(f"Saved diagnostics for layer: {layer_name}")

                diag_save_path = log_dir / "routing_diagnostics.npz"
                np.savez_compressed(diag_save_path, **npz_data)
                logging.info(f"Routing diagnostics saved to {diag_save_path}")
            except Exception as e:
                logging.error(f"Failed to save routing diagnostics: {e}")

        logging.info("\n--- Inference Complete ---")

if __name__ == "__main__":
    """
    Main entry point for the script. Parses command-line arguments, loads the model checkpoint,
    initiates the inference process. Optionally takes the argument for MoE router diagnostics.
    """
    parser = argparse.ArgumentParser(description="Evaluation script for vision models.")
    parser.add_argument(
        "-m", "--model_path", type=str, required=True,
        help="Path to the saved .pth model file (e.g., 'outputs/models/exp_name/best_model.pth')."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=None,
        help="Optional: Directory to save the output predictions/diagnostics. Defaults inferred from model path."
    )
    parser.add_argument(
        "--run_diagnostics", action="store_true",
        help="Capture and save router outputs (indices and weights) for MoE analysis."
    )
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model_path

    if args.output_dir is None:
        try:
            experiment_name = model_path.parent.name
            log_dir = PROJECT_ROOT / "outputs" / "logs" / experiment_name
        except IndexError:
             log_dir = PROJECT_ROOT / "outputs" / "logs" / "evaluate_run"
             logging.warning(f"Could not infer experiment name from {model_path}. Using default log dir: {log_dir}")
    else:
        log_dir = PROJECT_ROOT / args.output_dir

    evaluate(model_path, log_dir, run_diagnostics=args.run_diagnostics)