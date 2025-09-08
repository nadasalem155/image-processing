import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2

st.set_page_config(page_title="Image Editor", layout="centered")
st.title("🖼️ Simple Image Editor")

if 'base_image' not in st.session_state:
    st.session_state.base_image = None
if 'edited_image' not in st.session_state:
    st.session_state.edited_image = None

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.session_state.base_image = Image.open(uploaded_file)
    st.session_state.edited_image = st.session_state.base_image.copy()

if st.session_state.base_image:
    st.image(st.session_state.base_image, caption="Original Image", use_column_width=True)

    st.subheader("Adjustments")
    brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.01)
    contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.01)
    sharpness = st.slider("Sharpness", 0.0, 5.0, 1.0, 0.1)

    # ✅ Apply filters only if user actually changes a slider
    if brightness != 1.0 or contrast != 1.0 or sharpness != 1.0:
        temp_img = st.session_state.base_image.copy()

        if brightness != 1.0:
            temp_img = ImageEnhance.Brightness(temp_img).enhance(brightness)

        if contrast != 1.0:
            temp_img = ImageEnhance.Contrast(temp_img).enhance(contrast)

        if sharpness != 1.0:
            if sharpness > 2.0:
                temp_array = np.array(temp_img)
                temp_array = cv2.detailEnhance(
                    temp_array,
                    sigma_s=10 * (sharpness - 2.0) / 3.0,
                    sigma_r=0.15
                )
                temp_img = Image.fromarray(temp_array)
            temp_img = ImageEnhance.Sharpness(temp_img).enhance(sharpness)

        st.session_state.edited_image = temp_img
    
    if st.session_state.edited_image:
        st.image(st.session_state.edited_image, caption="Edited Image", use_column_width=True)