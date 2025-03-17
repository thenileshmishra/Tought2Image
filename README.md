
# **EEG-to-Image Reconstruction using Deep Learning**  

## **Overview**  
This project focuses on reconstructing images from EEG brain signals using advanced deep learning models, including GANs, Variational Autoencoders (VAEs), and Latent Diffusion Models (LDMs). The aim is to generate high-quality visual representations of perceived or imagined stimuli from EEG recordings.  

The implementation follows a structured pipeline for preprocessing EEG signals, feature extraction, and image reconstruction.  

---

## **Preprocessing Pipeline**  

The raw EEG signals undergo a series of preprocessing steps to enhance data quality and extract meaningful features:  

1. **Data Acquisition:**  
   - EEG data is recorded using high-density or portable EEG systems at sampling rates between 128 Hz and 1000 Hz.  
   - Signals are captured while participants are exposed to visual stimuli (perception) or asked to imagine specific images (imagination).  

2. **Filtering:**  
   - **Bandpass Filtering**: 1–50 Hz or 1–64 Hz to remove unwanted frequencies.  
   - **Notch Filtering**: Removes power-line interference at 50/60 Hz.  

3. **Artifact Removal:**  
   - **Independent Component Analysis (ICA)**: Identifies and removes artifacts caused by eye blinks, muscle movement, and noise.  
   - **Bad Trial/Channel Rejection**: Discards data with excessive noise or poor electrode connections.  

4. **Segmentation:**  
   - EEG signals are segmented into epochs relative to stimulus onset (e.g., −200 ms to 800 ms).  
   - Averaging over multiple trials improves the signal-to-noise ratio.  

5. **Normalization and Whitening:**  
   - Z-score normalization per channel to standardize signal amplitude.  
   - Whitening techniques are applied to remove correlations and enhance discriminative features.  

6. **Downsampling (if required):**  
   - To optimize performance, EEG data may be downsampled from 1000 Hz to 250 Hz or lower.  

---

## **Methodology**  

The project follows a deep learning-based approach to reconstruct images from EEG signals:  

### **1. Feature Extraction**  
- A deep neural network (DNN)-based encoder extracts meaningful representations from EEG signals.  
- Options include CNN-based models, LSTMs for temporal features, or contrastive learning methods.  

### **2. Image Generation Models**  
The extracted EEG features are used as input to different generative models:  

- **Generative Adversarial Networks (GANs)**:  
  - **NeuroGAN**: Uses an attention-based GAN to generate images from EEG features.  
  - **EEG2Image**: Implements a modified cGAN for robust reconstructions.  

- **Variational Autoencoders (VAEs)**:  
  - Used in some approaches for latent-space encoding and decoding.  

- **Latent Diffusion Models (LDMs)**:  
  - **ControlNet for EEG Conditioning**: EEG signals are mapped to diffusion model latents for high-fidelity reconstruction.  
  - **EEG-StyleGAN-ADA**: Uses EEG embeddings to generate photo-realistic images.  

- **Hybrid Models**:  
  - Some approaches combine GANs and diffusion models to balance image realism and EEG-to-image mapping accuracy.  

---

## **Installation & Requirements**  

### **1. Software Requirements**  
Ensure that the following dependencies are installed:  

- Python (>=3.8)  
- PyTorch (>=1.10)  
- TensorFlow (>=2.0) (optional)  
- NumPy  
- SciPy  
- OpenCV  
- Matplotlib  
- Scikit-learn  
- MNE (for EEG signal processing)  

### **2. Installation**  
Clone the repository:  
```bash
git clone https://github.com/thenileshmishra/Tought2Image
cd eeg-image-reconstruction
```
Install dependencies:  
```bash
pip install -r requirements.txt
```

---

## **Usage**  

### **1. EEG Data Preprocessing**  
Run the preprocessing script to filter, clean, and segment EEG data:  
```bash
python preprocess.py --input hugginFace "Alljoined/05_125" dataset --output data/EEG and data/metadata
```

### **2. Feature Extraction**  
Extract EEG features using a deep encoder model:  
```bash
python extract_features.py --input data/processed_eeg.npy --output data/eeg_features.npy
```

### **3. Image Generation**  
To generate images using GANs:  
```bash
python train_gan.py --input data/eeg_features.npy --output generated_images/
```
To use latent diffusion models:  
```bash
python train_ldm.py --input data/eeg_features.npy --output generated_images/
```

---

## **Results & Evaluation**  

- **Metrics Used:**  
  - Structural Similarity Index (SSIM)  
  - Frechet Inception Distance (FID)  
  - Peak Signal-to-Noise Ratio (PSNR)  

- **Qualitative Evaluation:**  
  - Generated images are visually compared with original images shown to the subjects.  
  - Examples of generated outputs are saved in `generated_images/`.  

---

## **Dataset Information**  

- The project uses publicly available EEG datasets from hugging face **Alljoined/05_125**.  
- Datasets include paired EEG-image_id (COCO dataset) samples for supervised training.  

---

## **Contributions**  
If you’d like to contribute, feel free to submit a pull request or open an issue.  

---

## **References**  
This project builds upon research from:  
1. NeuroGAN: Attention-Based GANs for EEG Image Reconstruction  
2. EEG2Image: Contrastive Learning for Brain-to-Image Synthesis  
3. Diffusion Model-Based EEG Image Generation  

---

## **License**  
This project is licensed under the MIT License.  

---

Let me know if you'd like any modifications! 🚀