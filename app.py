import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
import io
import tempfile
import os

# ------------------- Helper Functions ------------------- #
def cartoon_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def cartoon_colorful_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.addWeighted(color, 0.8, edges_colored, 0.2, 0)
    return cartoon

def hdr_filter(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

def pencil_sketch(img):
    gray, sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch

def oil_painting(img):
    return cv2.xphoto.oilPainting(img, 7, 1)

def sepia_filter(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    sepia = np.clip(sepia, 0, 255)
    return sepia.astype(np.uint8)

def negative_filter(img):
    return cv2.bitwise_not(img)

def grayscale_filter(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ------------------- Streamlit App ------------------- #
st.set_page_config(page_title="Image Editor", layout="wide")
st.title("🖼️ Advanced Image Editor")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(img)

    st.image(img_array, caption="Current Image", use_container_width=True)

    # ------------------- Filters ------------------- #
    st.sidebar.header("Filters")
    apply_filters = st.sidebar.multiselect(
        "Choose filters",
        ["Cartoon", "Cartoon Colorful", "HDR Enhanced", "Pencil Sketch",
         "Oil Painting", "Sepia", "Negative", "Grayscale", "Gaussian Blur"]
    )

    processed_img = img_array.copy()
    if apply_filters:
        for f in apply_filters:
            if f == "Cartoon":
                processed_img = cartoon_filter(processed_img)
            elif f == "Cartoon Colorful":
                processed_img = cartoon_colorful_filter(processed_img)
            elif f == "HDR Enhanced":
                processed_img = hdr_filter(processed_img)
            elif f == "Pencil Sketch":
                processed_img = pencil_sketch(processed_img)
            elif f == "Oil Painting":
                try:
                    processed_img = oil_painting(processed_img)
                except:
                    st.warning("⚠️ Oil Painting filter needs OpenCV contrib installed.")
            elif f == "Sepia":
                processed_img = sepia_filter(processed_img)
            elif f == "Negative":
                processed_img = negative_filter(processed_img)
            elif f == "Grayscale":
                processed_img = grayscale_filter(processed_img)
            elif f == "Gaussian Blur":
                processed_img = cv2.GaussianBlur(processed_img, (15, 15), 0)

    st.image(processed_img, caption="Filtered Image", use_container_width=True)

    # ------------------- Download ------------------- #
    result = Image.fromarray(processed_img)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.sidebar.download_button(
        label="💾 Download Result",
        data=byte_im,
        file_name="processed.png",
        mime="image/png"
    )