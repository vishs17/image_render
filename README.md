# 🔧 AI-Driven Denoising & Image Segmentation Enhancer

An advanced **Streamlit web app** for denoising, segmenting, and enhancing images using AI and computer vision.  
It combines **RIDNet**, **Watershed Segmentation**, and custom **HSV-based color enhancements** to deliver visually improved and analytically segmented results.

---

## 🚀 Features

- 🎨 **Auto Color Enhancement**
  - Uses CLAHE on the L channel in LAB color space.
  - Refines image contrast and color balance.

- 🧠 **AI-Based Denoising**
  - Leverages a deep learning model (RIDNet) to reduce noise while preserving edges.
  - Works on patchified images for high-resolution denoising.

- 🧫 **Watershed Segmentation**
  - Applies thresholding and morphological operations.
  - Segments the image and highlights object boundaries.

- 🔥 **Selective Color Enhancement**
  - Enhances red and orange tones using HSV adjustments.
  - Boosts visual clarity of key features in segmented regions.

- 💾 **Downloadable Enhanced Output**
  - Lets users download enhanced versions of the uploaded image.

---

## 🛠️ Tech Stack

- **Python**  
- **Streamlit** – UI & user interaction  
- **OpenCV** – Image processing  
- **NumPy** – Numerical operations  
- **TensorFlow / Keras** – Deep learning (RIDNet)  
- **Patchify / Unpatchify** – For image patch operations  
- **Matplotlib** – For plotting segmentation maps  
- **Pillow** – Image handling  

---

## 📦 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/vishs17/image_render.git
