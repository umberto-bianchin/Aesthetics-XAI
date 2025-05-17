import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import tensorflow as tf
from huggingface_hub import snapshot_download, from_pretrained_keras

model = from_pretrained_keras("alexanderkroner/MSI-Net")
hf_dir = snapshot_download(repo_id="alexanderkroner/MSI-Net")

# ----------------------------------------------------------------------------

def saliency(image):
    image = Image.open(image)
    # Predict saliency map
    saliency_output = predict_saliency(image, 1)  # assuming it outputs float64

    # Normalize if saliency_output is in [0, 1]
    if saliency_output.max() <= 1.0:
        saliency_output = saliency_output * 255.0

    # Convert to uint8
    saliency_output_uint8 = saliency_output.astype(np.uint8)

    # Convert saliency_output to grayscale
    saliency_gray = cv2.cvtColor(saliency_output_uint8, cv2.COLOR_RGB2GRAY)
    return saliency_gray
    
def predict_saliency(image, alpha):
    input_image = np.array(image.convert("RGB"), dtype=np.uint8)
    original_shape = input_image.shape[:2]
    target_shape = get_target_shape(original_shape)
    input_tensor, vertical_padding, horizontal_padding = preprocess_input(input_image, target_shape)
    output_tensor = model(input_tensor)["output"]
    saliency_map = postprocess_output(output_tensor, vertical_padding, horizontal_padding, original_shape)
    blended_image = alpha * saliency_map + (1 - alpha) * input_image / 255
    return blended_image

def get_target_shape(original_shape):
    original_aspect_ratio = original_shape[0] / original_shape[1]

    square_mode = abs(original_aspect_ratio - 1.0)
    landscape_mode = abs(original_aspect_ratio - 240 / 320)
    portrait_mode = abs(original_aspect_ratio - 320 / 240)

    best_mode = min(square_mode, landscape_mode, portrait_mode)

    if best_mode == square_mode:
        target_shape = (320, 320)
    elif best_mode == landscape_mode:
        target_shape = (240, 320)
    else:
        target_shape = (320, 240)

    return target_shape


def preprocess_input(input_image, target_shape):
    input_tensor = tf.expand_dims(input_image, axis=0)

    input_tensor = tf.image.resize(
        input_tensor, target_shape, preserve_aspect_ratio=True
    )

    vertical_padding = target_shape[0] - input_tensor.shape[1]
    horizontal_padding = target_shape[1] - input_tensor.shape[2]

    vertical_padding_1 = vertical_padding // 2
    vertical_padding_2 = vertical_padding - vertical_padding_1

    horizontal_padding_1 = horizontal_padding // 2
    horizontal_padding_2 = horizontal_padding - horizontal_padding_1

    input_tensor = tf.pad(
        input_tensor,
        [
            [0, 0],
            [vertical_padding_1, vertical_padding_2],
            [horizontal_padding_1, horizontal_padding_2],
            [0, 0],
        ],
    )

    return (
        input_tensor,
        [vertical_padding_1, vertical_padding_2],
        [horizontal_padding_1, horizontal_padding_2],
    )


def postprocess_output(
    output_tensor, vertical_padding, horizontal_padding, original_shape
):
    output_tensor = output_tensor[
        :,
        vertical_padding[0] : output_tensor.shape[1] - vertical_padding[1],
        horizontal_padding[0] : output_tensor.shape[2] - horizontal_padding[1],
        :,
    ]

    output_tensor = tf.image.resize(output_tensor, original_shape)

    output_array = output_tensor.numpy().squeeze()
    output_array = plt.cm.gray(output_array)[..., :3]

    return output_array

# ---------------------------------------------------------

# Places a 299x299 white square randomly with its center inside any blob's area in a binary image.
def place_white_square_randomly_in_blob(binary_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    if num_labels <= 1:
        raise ValueError("No blobs found in the image.")

    random_blob_label = random.randint(1, num_labels - 1)

    blob_pixels = np.where(labels == random_blob_label)

    random_index = random.randint(0, len(blob_pixels[0]) - 1)
    center_x = blob_pixels[1][random_index]
    center_y = blob_pixels[0][random_index]

    output = np.zeros_like(binary_image)

    half_size = 299 // 2
    top_left_x = max(center_x - half_size, 0)
    top_left_y = max(center_y - half_size, 0)
    bottom_right_x = min(center_x + half_size, output.shape[1])
    bottom_right_y = min(center_y + half_size, output.shape[0])

    output[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    return output

def apply_mask_to_image_and_crop(input_image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = input_image[y:y+h, x:x+w]
    return cropped_image

def extract_focus_patch(image_path, top_focus, samples):
    input_image = cv2.imread(image_path)
    saliency_map = saliency(image_path)
    _, binary = cv2.threshold(saliency_map, int(255 * (1-top_focus)), 255, cv2.THRESH_BINARY)
    os.makedirs(f"../top{int(top_focus*100)}", exist_ok=True)
    os.makedirs(f"../top{int(top_focus*100)}/{image_path.split('.')[0]}", exist_ok=True)

    for i in range(0, samples):
        print(i)
        result = place_white_square_randomly_in_blob(binary)
        masked_image = apply_mask_to_image_and_crop(input_image, result)
        cv2.imwrite(f"../top{int(top_focus*100)}/{image_path.split('.')[0]}/{i}.jpg", masked_image)


# -----------------------------------------------------------



def extract_patches(directory_path):
    files = os.listdir(directory_path)
    files = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]
    for file in files:
        print(file)
        extract_focus_patch(f"/content/EVA_together/{file}", 0.5, 5)

# usage
directory_path = '/content/EVA_together'  # replace with the right folder path
extract_patches(directory_path)
