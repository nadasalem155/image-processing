import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)

    st.image(img, caption="Current Image", use_container_width=True)

    # --- Text Overlay ---
    text = st.text_input("Enter text to overlay on image:")
    if text:
        img_array = cv2.putText(
            img_array.copy(),
            text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,  # bigger font size
            (255, 0, 0),
            4,
            cv2.LINE_AA,
        )
        st.image(img_array, caption="Image with Text", use_container_width=True)

    # --- Remove Background (simple white mask) ---
    if st.button("Remove Background"):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        result = cv2.bitwise_and(img_array, img_array, mask=mask)
        white_bg = np.ones_like(img_array, dtype=np.uint8) * 255
        mask_inv = cv2.bitwise_not(mask)
        final_img = cv2.bitwise_and(white_bg, white_bg, mask=mask_inv)
        final_img = cv2.add(final_img, result)
        st.image(final_img, caption="Background Removed", use_container_width=True)

    # --- Filters ---
    st.subheader("Apply Filters")
    filter_options = ["Cartoon", "Cartoon Colorful", "Blur", "Pencil Sketch", "HDR Enhanced"]
    apply_filters = st.multiselect("Select filters:", filter_options)

    if apply_filters:
        filtered_img = img_array.copy()
        for f in apply_filters:
            if f == "Cartoon":
                gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.adaptiveThreshold(gray, 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 9, 9)
                color = cv2.bilateralFilter(filtered_img, 9, 250, 250)
                filtered_img = cv2.bitwise_and(color, color, mask=edges)

            elif f == "Cartoon Colorful":
                gray = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
                gray = cv2.medianBlur(gray, 7)
                edges = cv2.Canny(gray, 50, 150)
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                color = cv2.bilateralFilter(filtered_img, 9, 300, 300)
                filtered_img = cv2.addWeighted(color, 0.8, edges_colored, 0.2, 0)

            elif f == "Blur":
                filtered_img = cv2.GaussianBlur(filtered_img, (15, 15), 0)

            elif f == "Pencil Sketch":
                gray, sketch = cv2.pencilSketch(filtered_img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
                filtered_img = sketch

            elif f == "HDR Enhanced":
                filtered_img = cv2.detailEnhance(filtered_img, sigma_s=12, sigma_r=0.15)

        st.image(filtered_img, caption="Filtered Image", use_container_width=True)