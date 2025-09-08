import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance

# ---- History ----
if "history" not in st.session_state:
    st.session_state.history = []

# ---- Helper to add to history ----
def add_to_history(img):
    st.session_state.history.append(img.copy())

# ---- Load Image ----
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    if "current_img" not in st.session_state:
        st.session_state.current_img = img.copy()
        add_to_history(img.copy())

    # ---- Filter Functions ----
    def brightness_filter(img, value):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1 + value)

    def contrast_filter(img, value):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1 + value)

    def sharpness_filter(img, value):
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(1 + value)

    def cartoon_filter(img, intensity=1.0):
        if intensity == 0:
            return img
        img_array = np.array(img)
        color = img_array.copy()
        num_bilateral = int(5 + 5 * intensity)
        for _ in range(num_bilateral):
            color = cv2.bilateralFilter(color, d=9, sigmaColor=9, sigmaSpace=7)

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        result = cv2.addWeighted(img_array, 1 - intensity, cartoon, intensity, 0)
        return Image.fromarray(result)

    def cartoon_colorful_filter(img, intensity=1.0):
        if intensity == 0:
            return img
        img_array = np.array(img)
        color = img_array.copy()
        num_bilateral = int(6 + 6 * intensity)
        for _ in range(num_bilateral):
            color = cv2.bilateralFilter(color, d=9, sigmaColor=12, sigmaSpace=9)

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        result = cv2.addWeighted(img_array, 1 - intensity, cartoon, intensity, 0)
        return Image.fromarray(result)

    def denoise_filter(img, strength):
        if strength == 0:
            return img
        img_array = np.array(img)
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, h=strength * 20, hColor=strength * 20, templateWindowSize=7, searchWindowSize=21)
        return Image.fromarray(denoised)

    # ---- Sliders ----
    st.sidebar.header("Adjustments")
    brightness = st.sidebar.slider("Brightness", -1.0, 1.0, 0.0, 0.01)
    contrast = st.sidebar.slider("Contrast", -1.0, 1.0, 0.0, 0.01)
    sharpness = st.sidebar.slider("Sharpness", -1.0, 2.0, 0.0, 0.01)
    denoise_strength = st.sidebar.slider("Denoise", 0.0, 1.0, 0.0, 0.01)
    cartoon_intensity = st.sidebar.slider("Cartoon", 0.0, 1.0, 0.0, 0.01)
    cartoon_colorful_intensity = st.sidebar.slider("Cartoon Colorful", 0.0, 1.0, 0.0, 0.01)

    # ---- Apply Filters ----
    temp_img = img.copy()
    temp_img = brightness_filter(temp_img, brightness)
    temp_img = contrast_filter(temp_img, contrast)
    temp_img = sharpness_filter(temp_img, sharpness)
    temp_img = denoise_filter(temp_img, denoise_strength)
    temp_img = cartoon_filter(temp_img, cartoon_intensity)
    temp_img = cartoon_colorful_filter(temp_img, cartoon_colorful_intensity)

    st.image(temp_img, caption="Preview", use_container_width=True)

    if st.button("Apply Denoise"):
        st.session_state.current_img = temp_img.copy()
        add_to_history(temp_img)
        st.success("Denoise Applied ✅")

    if st.button("Undo"):
        if len(st.session_state.history) > 1:
            st.session_state.history.pop()
            st.session_state.current_img = st.session_state.history[-1]
            st.success("Undone ⬅️")