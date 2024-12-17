import numpy as np
import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image

# Function to download the image
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

# Function to prepare the image
def prepare_image(img, target_size=(150, 150)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    img_array = np.array(img) / 255.0  # Rescale values
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model_2024_hairstyle_v2.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Download and prepare the image
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
image = download_image(url)
image_data = prepare_image(image)

# Perform inference
interpreter.set_tensor(input_index, image_data)
interpreter.invoke()
output = interpreter.get_tensor(output_index)

print(f"Model output: {output[0][0]:.3f}")
