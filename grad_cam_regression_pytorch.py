import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import models, transforms
from torchvision.models.inception import InceptionOutputs
from torchvision.models import Inception_V3_Weights
from torch.nn import functional as F
from PIL import Image
from pathlib import Path

# Mount Google Drive if using Colab
from google.colab import drive
drive.mount('/content/drive')

MODEL_PATH = '/content/drive/MyDrive/Colab_Notebooks/inception_multiout_final.pth'
IMG_PATH = '/content/drive/MyDrive/Colab_Notebooks/EVA_together/*.jpg'
SAVE_DIR = '/content/drive/MyDrive/Colab_Notebooks/Results'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiOutputInception(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        base = models.inception_v3(weights=weights, aux_logits=True)
        base.AuxLogits = None  # Remove auxiliary classifier
        base.fc = torch.nn.Identity()
        self.base = base
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.base(x)
        return self.head(x)

# Load weights
model = MultiOutputInception().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

target_layer = model.base.Mixed_7c

""" Check states
state_dict = torch.load(MODEL_PATH)
for k, v in state_dict.items():
    print(k, v.shape)
"""

# Transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# GradCAM heatmap creation
def make_gradcam_heatmap(img_tensor, model, target_layer, score_index):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(img_tensor)
    loss = output[0, score_index]
    loss.backward()

    activ = activations[0].squeeze(0)  # (C, H, W)
    grads = gradients[0].squeeze(0)    # (C, H, W)

    weights = grads.mean(dim=(1, 2))                            # (C,)
    heatmap = torch.sum(weights[:, None, None] * activ, dim=0)  # (H, W)
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().cpu().numpy()

    handle_fwd.remove()
    handle_bwd.remove()

    return heatmap

# Image preprocessing
def load_and_preprocess_image(img_path):
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, original_img

# Heatmap superimposition on original image
def superimpose_heatmap(heatmap, original_image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = 1 - heatmap  # flip color importance
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed

# Run gradCAM on images
def run_gradcam_on_image(img_path, save_dir=SAVE_DIR):
    image_basename = Path(img_path).stem
    image_folder = os.path.join(save_dir, image_basename)
    os.makedirs(image_folder, exist_ok=True)

    img_tensor, original_img = load_and_preprocess_image(img_path)
    score_names = ['total', 'difficulty', 'visual', 'composition', 'quality', 'semantic']

    for i, score in enumerate(score_names):
        heatmap = make_gradcam_heatmap(img_tensor, model, target_layer, score_index=i)
        cam = superimpose_heatmap(heatmap, original_img)
        save_path = os.path.join(image_folder, f"{score}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))
        print(f"Saved: {save_path}")

# Run Grad-CAM over all images
image_paths = glob.glob(IMG_PATH)
for image_path in image_paths:
    run_gradcam_on_image(image_path)
