# Aesthetics-XAI: Predicting Image Aesthetic Quality with Human Saliency–Guided Patches

## Authors

- [@umberto-bianchin](https://github.com/umberto-bianchin)
- [@Mateusz Miroslaw Lis](https://github.com/supergeniodelmale)
- [@Arjun Jassal](https://github.com/Faldrus)

---

## Overview

This project implements a two-stage deep learning pipeline for image aesthetic quality assessment. It first extracts human-saliency–guided patches from images and then trains a multi-output regression model to predict six aesthetic scores:

* **Total** (overall aesthetic quality)
* **Difficulty**
* **Visual** appeal
* **Composition**
* **Quality**
* **Semantic** coherence

By leveraging both global and fine-grained local cues, our approach achieves improved performance over full-image analysis and provides explainability via Grad-CAM.

## Repository Structure

```
Aesthetics-XAI/
├── extract_saliency_patches.py   # Patch extraction using MSI-Net saliency
├── inception_train.py            # Training multi-output InceptionV3 regression
├── model_evaluate.py             # Evaluation on full images and patches (ONNX)
├── grad_cam_regression.py        # Grad-CAM visualizations for each score
├── requirements.txt              # Python dependencies
├── data/                         # (to be created) dataset directory
│   ├── images/                   # Original EVA images
│   └── votes_filtered.csv        # Filtered votes CSV from EVA dataset
└── README.md                     
```

## Requirements

## Requirements
- Python 3.7+
- tensorflow
- numpy
- pandas
- scikit-learn
- Pillow
- opencv-python
- matplotlib
- huggingface_hub
- pytorch
- torchvision
- torchaudio
- tqdm
- libjpeg-turbo
- onnx
- onnxruntime

Install via:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. **EVA dataset**: Download images and filtered votes CSV from the EVA dataset repository:

   ```bash
   git clone https://github.com/umberto-bianchin/Aesthetics-XAI.git
   mkdir datasets && cd Aesthetics-XAI/datasets
   # Place EVA images in `datasets/eva-dataset-master/images/`
   # Place `votes_filtered.csv` in `datasets/eva-dataset-master/data`
   ```
2. **Directory paths**: Update the default paths in the scripts if needed:

## Usage

### 1. Extract Saliency-Guided Patches

```bash
python extract_saliency_patches.py
```

* Generates five 299×299 patches per image, stored in `topXX/` subfolders.

### 2. Train the Regression Model

```bash
python inception_train.py
```

* Stage 1: Train dense head on frozen InceptionV3 backbone.
* Stage 2: Fine-tune top convolutional layers.
* Checkpoints and TensorBoard logs are saved in the working directory.

### 3. Evaluate Model Performance

```bash
python model_evaluate.py
```

* Computes MAE and R² on full images and aggregated patches.
* Outputs metrics tables and saves detailed results CSV.

### 4. Generate Grad-CAM Visualizations

```bash
python grad_cam_regression.py
```

* Produces heatmaps for each of the six scores.
* Compares model saliency with human saliency maps.
* Saves overlay images to the specified `SAVE_DIR`.

## Results

* **Full-Image**: MAE ≈ 0.58, R² ≈ 0.48 on total score.
* **Patch-Aggregated**: MAE ≈ 0.46, R² ≈ 0.68 on total score.
* Significant gains on visual, composition, and semantic dimensions.

## Explainability (XAI)

We adapt Grad-CAM for regression by computing gradients of each output neuron w\.r.t. the final convolutional feature maps. Visualizations reveal that the model focuses on similar regions as human saliency, validating its interpretability.

---