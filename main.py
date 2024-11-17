from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import cv2
import numpy as np

import os
import torch
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model globally
model = smp.from_pretrained("trained_model")
model.eval()
model.to(device)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

denorm = A.Normalize(
    mean=[-m / s for m, s in zip(mean, std)],
    std=[1.0 / s for s in std],
    always_apply=True,
    max_pixel_value=1.0
)
tt = ToTensorV2()

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace this with your actual image segmentation model
def segment_image(image_bytes):
    """
    This function should perform image segmentation on the input image bytes
    and return the segmented image as bytes.

    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Placeholder segmentation logic: Apply a simple threshold to create a mask
    train_transform = A.Compose(
        [
            A.Resize(1024, 1024),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    dt = train_transform(image=img_np, mask=img_np)
    img = dt['image']
    width = img.shape[1]
    height = img.shape[2]

    img = img.to(device)

    with torch.no_grad():
      output = model(img.unsqueeze(0))

    output_mask = output.argmax(axis=1)
    
    # Create a color map for visualization (background is black, class of interest is red)
    color_map = np.zeros((1024, 1024, 3), dtype=np.float32)
    color_map[output_mask[0].cpu().numpy() == 1] = [1.0, 0.0, 0.0]  # Red color for the class of interest

    # Overlay the mask on the input image (input_image is in shape (3, 1024, 1024), so we need to transpose it)
    input_image = img.permute(1,2,0).cpu().numpy().copy()  # Shape: (1024, 1024, 3)

    # Blend the original image and mask using transparency
    alpha = 0.4  # Transparency factor
    overlayed_image = cv2.addWeighted(input_image, 1 - alpha, color_map, alpha, 0)

    # Create a binary mask for the class of interest
    binary_mask = (output_mask[0].cpu().numpy() == 1).astype(np.uint8)  # Class 1 is the class of interest

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the input image (green borders with thickness 2)
    border_color = (0, 255, 0)  # Green color for the border
    _ = cv2.drawContours(input_image, contours, -1, border_color, 2)

    plt.figure(figsize=(15,15))
    plt.subplot(1,4,1)
    plt.imshow(img.permute(1,2,0).cpu().numpy())
    plt.title('Input image')
    plt.axis(False)
    plt.subplot(1,4,2)
    plt.imshow(input_image)
    plt.title('Segmentation Border')
    plt.axis(False)
    plt.subplot(1,4,3)
    plt.imshow(overlayed_image)
    plt.title('Segmentation Mask Overlay')
    plt.axis(False)
    plt.savefig('./result.jpg')

    # Convert the segmented image to bytes
    is_success, buffer = cv2.imencode(".jpg", cv2.imread('./result.jpg', cv2.IMREAD_COLOR))
    segmented_image_bytes = BytesIO(buffer)
    return segmented_image_bytes



@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        segmented_image_bytes = segment_image(contents)
        return StreamingResponse(segmented_image_bytes, media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def root():
    return {"message": "Hello World"}


# # Run the server using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
