import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
from patchify import patchify, unpatchify
from PIL import Image
from pathlib import Path
import os
import matplotlib.pyplot as plt



st.title("AI Driven Denoising & Image Segmantation Enhancer")
st.write("Upload an image to Process")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def auto_adjust(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
   
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
   
    lab_enhanced = cv2.merge((l, a, b))
    contrast_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
  
    adjusted_img = cv2.convertScaleAbs(contrast_enhanced, alpha=1, beta=-10)  
    final_img_rgb = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_img_rgb)

@st.cache_resource
def get_model():
    model = tf.keras.models.load_model('RIDNet.h5', compile=False)
    return model

def create_patches(img, patch_size):
    return patchify(img, (patch_size, patch_size, 3), step=patch_size)

def denoise_image(img, target_size):
    st.text("Denoising in process...")
    state = st.text('Please wait while the model denoises the image...')
    progress_bar = st.progress(0)
    start = time.time()
    
    model = get_model()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1024, 1024))
    img = img.astype("float32") / 255.0

    img_patches = create_patches(img, 256)
    progress_bar.progress(30)
    img_patches = [img_patches[i][j][0] for i in range(4) for j in range(4)]
    img_patches = np.array(img_patches)

    pred_img = model.predict(img_patches)
    progress_bar.progress(70)
    pred_img = np.reshape(pred_img, (4, 4, 1, 256, 256, 3))
    pred_img = unpatchify(pred_img, img.shape)
    
   
    pred_img = cv2.resize(pred_img, target_size) 
    end = time.time()
    
   
    st.image(pred_img, caption="Denoised Image", use_column_width=True)
    st.write('Time taken for prediction:', f"{round(end - start, 3)} seconds")
    
    progress_bar.progress(100)
    state.text('Completed!')
    progress_bar.empty()
    return pred_img


def watershed_segmentation(image):
    
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image * 255)  
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    st.write("Threshold limit: " + str(ret))

    
    col1, col2 = st.columns(2)

    with col1:
        st.image(thresh, channels="GRAY", caption="Otsu's Binarization", use_column_width=True)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    fig = plt.figure(figsize=(8, 4))
    plt.subplot(131)
    plt.imshow(sure_bg, cmap='gray')
    plt.title('Sure Background (Dilated)')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(sure_fg, cmap='gray')
    plt.title('Sure Foreground (Eroded)')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(unknown, cmap='gray')
    plt.title('Unknown Region')
    plt.axis('off')

    
    with col2:
        st.pyplot(fig)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    
    output_image = image.copy()
    output_image[markers == -1] = [0, 255, 0]  

    return output_image  

def enhance_red_orange_contrast(image):
    
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])

    
    red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    orange_mask = cv2.inRange(hsv_img, lower_orange, upper_orange)

    
    mask = cv2.bitwise_or(red_mask, orange_mask)

    
    hsv_img[:, :, 1] = np.where(mask > 0, np.clip(hsv_img[:, :, 1] * 1.2, 0, 255), hsv_img[:, :, 1])  
    hsv_img[:, :, 2] = np.where(mask > 0, np.clip(hsv_img[:, :, 2] * 1.1, 0, 255), hsv_img[:, :, 2]) 
    
    enhanced_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return enhanced_img

def enhance_image_colors(original_image):
    enhanced_image = enhance_red_orange_contrast(original_image)
    return enhanced_image

if uploaded_file is not None:
    
    original_image = Image.open(uploaded_file)
    image_np = np.array(original_image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    enhanced_image = auto_adjust(image_cv)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
    st.write("Enhancement complete.")
    
    
    enhanced_image_path = "enhanced_image.png"
    enhanced_image.save(enhanced_image_path)
    with open(enhanced_image_path, "rb") as file:
        st.download_button(label="Download Enhanced Image", data=file, file_name="enhanced_image.png", mime="image/png")
    
    
    uploaded_filename = uploaded_file.name
    noisy_image_path = Path(IMAGE_FOLDER) / uploaded_filename
    
    if noisy_image_path.is_file():
       
        st.write("Proceeding with denoising...")
        nsy_img = cv2.imread(str(noisy_image_path))
        denoised_image = denoise_image(nsy_img, original_image.size)
        
       
        segmented_image = watershed_segmentation(denoised_image)
        enhanced_segmented_image = enhance_image_colors(denoised_image)
        
        col3, col4 = st.columns(2)
        with col3:
            st.image(segmented_image, caption="Segmented Image", use_column_width=True)
        with col4:
            st.image(enhanced_segmented_image, caption="Enhanced Segmented Image", use_column_width=True)
