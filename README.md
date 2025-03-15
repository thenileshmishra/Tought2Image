# Thought2Image Project

This project implements a Dual-Encoder Joint VAE (or VQ-VAE) with CLIP Guidance for reconstructing images from EEG signals.

## Folder Structure


---

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
