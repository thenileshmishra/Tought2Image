# Thought2Image Project

This project implements a Dual-Encoder Joint VAE (or VQ-VAE) with CLIP Guidance for reconstructing images from EEG signals.

## Folder Structure


Thought2Image/
├── data/
│   ├── raw/
│   │   ├── EEG/                   # Raw EEG recordings (e.g., CSV, EDF files)
│   │   └── images/                # Raw image files (e.g., JPEG, PNG)
│   ├── processed/
│   │   ├── EEG/                   # Preprocessed EEG data (e.g., numpy arrays, TFRecords)
│   │   └── images/                # Preprocessed images (resized, normalized)
│   └── annotations/               # Labels, metadata, or text prompts for CLIP guidance
├── notebooks/                     # Jupyter notebooks for EDA, preprocessing, and experiments
│   ├── data_exploration.ipynb
│   ├── preprocessing.ipynb
│   └── model_evaluation.ipynb
├── src/                           # Source code for model implementation and utilities
│   ├── __init__.py
│   ├── config.py                  # Configuration files (hyperparameters, paths)
│   ├── data/                      
│   │   ├── dataset.py             # Dataset classes for EEG & images
│   │   └── dataloader.py          # Data loading utilities
│   ├── models/                    
│   │   ├── eeg_encoder.py         # EEG encoder implementation
│   │   ├── image_encoder.py       # Image encoder (for VAE/VQ-VAE)
│   │   ├── decoder.py             # Decoder architecture for image reconstruction
│   │   ├── joint_vae.py           # Joint VAE/VQ-VAE that fuses EEG & image latents
│   │   ├── clip_guidance.py       # Functions for extracting CLIP embeddings and computing CLIP loss
│   │   └── loss.py                # Combined loss functions (reconstruction, KL, alignment, CLIP)
│   ├── train.py                   # Training script for the full pipeline
│   ├── inference.py               # Inference script for generating images from EEG only
│   └── utils.py                   # Helper functions (logging, metrics, visualization)
├── experiments/                   # Files related to experimental runs
│   ├── logs/                      # Training logs, tensorboard files, etc.
│   ├── checkpoints/               # Saved model weights and checkpoints
│   └── results/                   # Generated images, evaluation metrics, and reports
├── scripts/                       # Shell scripts for running experiments
│   ├── run_training.sh
│   └── run_inference.sh
├── requirements.txt               # Python dependencies list
├── README.md                      # Project overview and setup instructions
└── LICENSE                        # License file

### 2. **requirements.txt**

This text file lists all the Python packages needed for your project. An example might look like:


Adjust versions as needed.

---

### 3. **Shell Scripts**

#### **scripts/run_training.sh**

This file automates the training process. For example:

```bash
#!/bin/bash
# Activate virtual environment if necessary
source ../env/bin/activate

# Run the training script with configuration file
python ../src/train.py --config ../src/config.yaml


data:
  eeg_path: "data/processed/EEG/"
  image_path: "data/processed/images/"
  annotations_path: "data/annotations/"

training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.0001
  kl_weight: 0.01
  align_weight: 1.0
  clip_weight: 0.1

model:
  eeg_encoder_dim: 256
  latent_dim: 256
  image_encoder: "VQ-VAE"
  use_clip: true

output:
  checkpoint_dir: "experiments/checkpoints/"
  log_dir: "experiments/logs/"
  results_dir: "experiments/results/"

