import cv2
import numpy as np
import os
def apply_clahe(image):
    """ ??????????????????????????????????????? """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_img = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)


# **????????????**
D = os.listdir('dataset\oxford-iiit-pet\images')
out_dir = 'dataset\oxford-iiit-pet\preimages'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for file in D:
    image = cv2.imread(os.path.join(D,file))

    # **Z-score ?????????**
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

    # **?????????** (??????????????????)
    denoised_image = cv2.bilateralFilter(normalized_image, 5, 25, 25)

    # **?????????????????? (CLAHE)**
    enhanced_image = apply_clahe(denoised_image)


    # **????????????**
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Normalize Image',normalized_image)
    # cv2.imshow('Denoised Image', denoised_image)
    # cv2.imshow('Enhanced Image (CLAHE)', enhanced_image)
    # cv2.imshow('hightlight protected:',protected_image)
    # # cv2.imshow('Sharpened Image (Final)', sharpened_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(out_dir,file),enhanced_image)

# **??????????????????**
# print(f"????????????: {avg_brightness:.2f}, ????????????: {final_weight:.2f}")
