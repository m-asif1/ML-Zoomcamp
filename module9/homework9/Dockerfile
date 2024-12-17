FROM agrigorev/model-2024-hairstyle:v3

# Install required libraries
RUN pip install numpy==1.23.1 pillow

# Copy your Lambda function code into the image
COPY lambda_function.py .

# Install TensorFlow Lite runtime
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

# Command to run the Lambda function locally
CMD ["python", "lambda_function.py"]