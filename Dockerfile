# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    segmentation-models-pytorch \
    albumentations \
    python-multipart \
    nest-asyncio \
    pyngrok \
    matplotlib \
    torchvision \
    opencv-python-headless \
    torch

# Copy the application code to the container
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
