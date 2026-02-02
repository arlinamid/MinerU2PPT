import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import win32com.client
from PIL import Image

def ppt_to_images(ppt_path, output_dir):
    """Convert PPT slides to images using Windows COM."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    powerpoint = win32com.client.Dispatch("PowerPoint.Application")
    presentation = powerpoint.Presentations.Open(os.path.abspath(ppt_path), WithWindow=False)

    image_paths = []
    for i, slide in enumerate(presentation.Slides):
        img_path = os.path.join(output_dir, f"slide_{i+1}.png")
        slide.Export(os.path.abspath(img_path), "PNG")
        image_paths.append(img_path)

    presentation.Close()
    powerpoint.Quit()
    return image_paths

def calculate_similarity(img1_path, img2_path):
    """Calculate SSIM between two images."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Resize img2 to match img1
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    return score, diff

def create_comparison(img1_path, img2_path, output_path, score):
    """Create side-by-side comparison."""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    combined = np.hstack((img1, img2))

    # Add text for score
    cv2.putText(combined, f"SSIM: {score:.4f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(output_path, combined)
