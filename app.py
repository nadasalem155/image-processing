import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import io
from streamlit_cropper import st_cropper

# ---- Helper Function ----
def get_mobile_dimensions(img):
    width, height = img.size
    max_width = 400
    if width > max_width:
        ratio = max_width / float(width)
        return max_width, int(height * ratio)
    return width, height

st.set_page_config(page_title="Image Editor", layout="wide")
st.title("🖼 Image Editing App")

# ---- Upload Section ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ---- Sidebar: Editing Tools ----
st.sidebar.header("🛠 Editing Tools")
denoise_strength = st.sidebar.slider("Denoise Strength 🧹", 0, 30, 0, 1)
rotate_90 = st.sidebar.checkbox("Rotate 90° 🔄")
apply_crop = st.sidebar.checkbox("✂ Crop")
apply_text = st.sidebar.checkbox("📝 Add Text")

# ---- Sidebar: Adjustments ----
st.sidebar.header("⚙ Adjustments")
brightness = st.sidebar.slider("Brightness ☀", -1.0, 1.0, 0.0, 0.01)
contrast = st.sidebar.slider("Contrast 🎚", -1.0, 1.0, 0.0, 0.01)
sharpness = st.sidebar.slider("Sharpness 🔪", -2.0, 3.0, 0.0, 0.01)

if uploaded_file:
    uploaded_image = Image.open(uploaded_file).convert("RGB")

    if ("base_image" not in st.session_state) or ("uploaded_file_name" not in st.session_state) or (st.session_state.uploaded_file_name != uploaded_file.name):
        st.session_state.base_image = uploaded_image.copy()
        st.session_state.edited_image = uploaded_image.copy()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.history = [uploaded_image.copy()]

    # ✅ Start from the latest base image (not always the original)
    img = st.session_state.base_image.copy()

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_png = Image.open(buf)
    final_width, final_height = get_mobile_dimensions(img)

    # ---- Cropping ----
    if apply_crop:
        st.write("✂ Drag the box to crop the image")
        cropped_img = st_cropper(img_png, realtime_update=True, box_color="red", aspect_ratio=None)
        if st.button("Apply Crop"):
            img = cropped_img
            st.session_state.base_image = img.copy()  # update base image after crop
            st.session_state.history.append(img.copy())
            st.success("Crop applied!")

    # ---- Apply Denoise ----
    if denoise_strength > 0:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(
            cv_img, None,
            h=denoise_strength, hColor=denoise_strength,
            templateWindowSize=7, searchWindowSize=21
        )
        img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

    # ---- Adjustments ----
    if brightness != 0.0:
        img = ImageEnhance.Brightness(img).enhance(1.0 + brightness)

    if contrast != 0.0:
        img = ImageEnhance.Contrast(img).enhance(1.0 + contrast)

    if sharpness != 0.0:
        if sharpness > 2.0:
            temp_array = np.array(img)
            temp_array = cv2.detailEnhance(
                temp_array,
                sigma_s=10 * ((sharpness - 2.0) / 3.0),
                sigma_r=0.15
            )
            img = Image.fromarray(temp_array)
        img = ImageEnhance.Sharpness(img).enhance(1.0 + sharpness)

    # Save to session_state so UI updates
    st.session_state.edited_image = img

    # ---- Rotation ----
    if rotate_90:
        if st.button("Apply 90° Rotation 🔄"):
            img = img.rotate(90, expand=True)
            st.session_state.base_image = img.copy()  # update base image after rotation
            st.session_state.history.append(img.copy())
            st.success("Rotation applied!")

    # Display image
    st.image(st.session_state.edited_image, use_container_width=True)