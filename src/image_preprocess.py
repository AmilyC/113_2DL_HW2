import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerrate broken images


def apply_clahe(image):
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_img = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)


D = 'dataset/oxford-iiit-pet/images'

out_dir = os.path.join('dataset', 'oxford-iiit-pet', 'preimages')

os.makedirs(out_dir, exist_ok=True)


for file in tqdm(os.listdir(D)) :
    if not file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        print(f"skip not images file: {file}")
        continue

    image_path = os.path.join(D, file)
    
    image = cv2.imread(image_path)
    
    if image is None:
       print(f"skip broken images file: {file}")
       continue
       
      
    
    

    # **Z-score **
    normalized_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        channel = image[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        normalized_channel = (channel - mean) / (std + 1e-8)
        normalized_channel = ((normalized_channel - np.min(normalized_channel)) /
                            (np.max(normalized_channel) - np.min(normalized_channel)) * 255)
        normalized_image[:, :, i] = normalized_channel

    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)

  
    denoised_image = cv2.bilateralFilter(normalized_image, 5, 25, 25)

 
    enhanced_image = apply_clahe(denoised_image)


 
    cv2.imwrite(os.path.join(out_dir,file),enhanced_image)


