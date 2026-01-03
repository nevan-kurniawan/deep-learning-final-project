# MoE-LoRA Texture Classification
This repository implements a **Mixture of Experts (MoE)** architecture combined with **Low-Rank Adaptation (LoRA)** to efficiently fine-tune Vision Transformers (DeiT-Small and ViT-Small) on the Describable Textures Dataset (DTD).

![MoE Architecture](docs/architecture_diagram.png)
*(Note: Replace with your actual architecture diagram or delete this line)*

The project investigates whether dynamic routing (MoE) combined with parameter-efficient fine-tuning (LoRA) can outperform static LoRA and Linear Probing in fine-grained texture classification tasks while maintaining parameter efficiency.

## Repository Structure

```
.
├── app/                  # Streamlit web application for inference
├── config/               # YAML configuration files for training
├── data/                 # Dataset storage (downloaded automatically)
├── notebooks/            # Analysis and EDA
│   ├── EDA.ipynb         # Data distribution and sample visualization
│   └── analysis.ipynb    # Training metrics, loss curves, and router diagnostics
├── outputs/              # Logs and saved models
├── scripts/              # Batch execution scripts (Windows)
├── src/                  # Core source code
│   ├── data_loader.py    # DTD dataset loading and transformations
│   ├── evaluate.py       # Inference and routing diagnostics
│   ├── model.py          # Custom MoE-LoRA layer and Router implementation
│   ├── train.py          # Training loop, logging, and checkpointing
│   └── utils.py          # Helper functions (seed, logging, GFLOPs)
└── README.md

```

## Methodology

### Mixture of Experts with LoRA

Standard LoRA injects trainable low-rank matrices into attention layers. This project modifies the standard ViT/DeiT attention blocks by replacing them with a custom `MoE_LoRA` layer (`src/model.py`).

* **Router:** A Top-k gating network () that dynamically selects experts based on input tokens.
* **Experts:** A `BatchedLoRA` module containing  sets of LoRA adapters (A and B matrices).
* **Load Balancing:** An auxiliary loss is computed during training to prevent expert collapse and ensure uniform usage of experts.

### Baselines

The proposed method is compared against:

1. **Full Fine-Tuning:** Updating all parameters.
2. **Linear Probe:** Training only the classification head.
3. **Static LoRA:** Standard Low-Rank Adaptation without routing.

## Installation

1. **Clone the repository:**
```
git clone [https://github.com/nevan-kurniawan/deep-learning-final-project.git](https://github.com/nevan-kurniawan/deep-learning-final-project.git)
cd deep-learning-final-project

```


2. **Environment Setup:**
Create a virtual environment and install dependencies.
```
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Install Root Dependencies (Training & Research)
pip install -r requirements.txt

```



## Experiments & Reproducibility

The project relies on configuration files located in `config/`. The DTD dataset will be downloaded automatically by `src/data_loader.py` upon the first run.

### Running Training

To train a specific model configuration:

```
python -m src.train --config config/deit_moe_lora.yaml

```

### Running Evaluation

To evaluate a trained model and generate predictions:

```
python -m src.evaluate --model_path outputs/models/deit_moe_lora/best_deit_moe_lora_model.pth

```

*Add `--run_diagnostics` to export router decisions for analysis.*

### Batch Reproduction (Windows)

Scripts are provided to reproduce the full suite of experiments (8 runs total):

* **Training:** `scripts/run_experiment.bat` (Runs all 4 methods across both backbones).
* **Evaluation:** `scripts/run_evaluation.bat` (Generates metrics for all trained models).

## Analysis

Use the notebooks in `notebooks/` to visualize results:

* `EDA.ipynb`: Inspects class balance and intra-class variance of the DTD dataset.
* `analysis.ipynb`: Loads training logs and evaluation CSVs to generate:
* Accuracy vs. Parameter Efficiency plots.
* Loss curves (Training/Validation/Auxiliary).
* Expert Routing Frequency histograms (visualizing expert specialization).



## Interactive Demo

A Streamlit application is included to demonstrate real-time inference using the trained MoE-LoRA model.

1. **Install App Dependencies:**
```bash
pip install -r app/requirements.txt

```


2. **Run the App:**
```bash
streamlit run app/app.py

```



The app features:

* **Image Upload:** Classify static images.
* **Live Prediction:** Real-time texture classification via webcam using `streamlit-webrtc`.