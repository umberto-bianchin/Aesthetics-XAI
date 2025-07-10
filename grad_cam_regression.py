import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from google.colab import drive
drive.mount('/content/drive')

MODEL_PATH = '/content/drive/MyDrive/Colab_Notebooks/inception_multiout_final.keras'
IMG_PATH = '/content/drive/MyDrive/Colab_Notebooks/EVA_together/*.jpg'
SAVE_DIR = '/content/drive/MyDrive/Colab_Notebooks/Results'

# Load model
model = load_model(MODEL_PATH)

# Get the last layer
last_conv_layer_name = 'mixed10'

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, score_index):

  # Create a sub model
  grad_model = tf.keras.models.Model(model.input, [model.get_layer(last_conv_layer_name).output, model.output])    # ([model.inputs], [conv_output, predictions])

  # Forward pass with GradientTape
  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model([img_array])
    loss = predictions[:, score_index]    # Select the target score

  # Compute the gradient of the target score
  grads = tape.gradient(loss, conv_outputs)

  # Global average pooling of the gradient to get the importance vector
  pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))

  # Reshape from (1, H, W, C) to (H, W, C)
  conv_outputs = conv_outputs[0]
  # Multiply each channel in the conv output by its corresponding importance weight and get a 2D (H, W)
  heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis = -1)

  # Applies a ReLU to only keep positive values
  heatmap = tf.maximum(heatmap, 0)
  # Normalize the heatmap 
  heatmap /= tf.reduce_max(heatmap)

  return heatmap.numpy()

def superimpose_heatmap(heatmap, original_image, alpha = 0.4):
  # Resize the 2D heatmap to the original image
  heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))  #OpenCV uses (width, height)

  # Convert the normalized heatmap from [0, 1] to [0, 255] for coloring
  heatmap = 1 - heatmap # Correct the heatmap because colors are inverted
  heatmap = np.uint8(255 * heatmap)

  # Applies colormap to greyscale image and get 3 channel colored heatmap (red = high importance, blue = low importance)
  heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

  # Superimpose heatmap on original image
  # output = (1-alpha)*image + alpha*heatmap where alpha is the opacity
  superimposed = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0) 

  return superimposed

def load_and_preprocess_image(img_path, target_size = (299, 299)):
  # Load original image for visualization in RGB
  original_img = cv2.imread(img_path)
  original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

  # Load + resize for model input
  img = image.load_img(img_path, target_size = target_size)
  # Convert image to numpy array of shape (299, 299, 3)
  img_array = image.img_to_array(img)
  # Expand input because Keras expects input size (1, 299, 299, 3)
  img_array_exp = np.expand_dims(img_array, axis = 0)
  preprocessed = preprocess_input(img_array_exp)

  # Return a tuple with
  # Scaled pixels from [0, 255] to [-1, 1] 
  # Original image unprocessed of shape (299, 299, 3) for visualization 
  return preprocessed, original_img

def run_gradcam_on_image(img_path, save_dir='/content/drive/MyDrive/Colab_Notebooks/Results'):
  # Extract base image name
  image_basename = os.path.splitext(os.path.basename(img_path))[0]

  # Create folder Results/image_basename
  image_folder = os.path.join(save_dir, image_basename)
  os.makedirs(image_folder, exist_ok=True)
  
  preprocessed_img, original_img = load_and_preprocess_image(img_path)

  # List of labes of scores
  score_names = ['total', 'difficulty', 'visual', 'composition', 'quality', 'semantic']
  
  for i, score in enumerate(score_names):
    heatmap = make_gradcam_heatmap(preprocessed_img, model, last_conv_layer_name, score_index = i)
    cam = superimpose_heatmap(heatmap, original_img)

    # Save each heatmap overlay image in Google Drive folder
    save_path = os.path.join(image_folder, f"{image_basename}_{score}.jpg")
    cv2.imwrite(save_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM of '{image_basename}' for '{score}' in: {save_path}")

image_paths = glob.glob(IMG_PATH)
for image_path in image_paths:
  run_gradcam_on_image(image_path)
