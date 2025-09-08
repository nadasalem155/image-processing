import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2
import io
from streamlit_cropper import st_cropper

# ---- Filter Functions ----
# (keep all your filter functions as you wroteهم)

# ---- Page config ----
st.set_page_config(page_title="📸🎨🖌 Image Editing App", layout="centered")
st.title("📸🎨🖌 Image Editing App – Easy & Fun Photo Editing")

def get_mobile_dimensions(pil_img, max_width=350):
    aspect_ratio = pil_img.height / pil_img.width
    width = min(pil_img.width, max_width)
    height = int(width * aspect_ratio)
    return width, height

# ---- Sidebar: Adjustments ----
st.sidebar.header("⚙ Adjustments")
brightness = st.sidebar.slider("Brightness ☀", 0.0, 2.0, 1.0, 0.01)
contrast = st.sidebar.slider("Contrast 🎚", 0.0, 2.0, 1.0, 0.01)
sharpness = st.sidebar.slider("Sharpness 🔪", 0.0, 5.0, 1.0, 0.01)

# ---- Sidebar: Filters & Effects ----
st.sidebar.header("🎨 Filters & Effects")
filter_options = ["Grayscale", "Sepia", "Blur", "Cartoon", "Cartoon Colorful", "HDR Enhanced"]
apply_filters = st.sidebar.multiselect("Filters 🎭", filter_options)

filter_intensities = {}
for f in filter_options:
    if f in apply_filters:
        filter_intensities[f] = st.sidebar.slider(
            f"Intensity of {f} (%)",
            0.0, 1.0, 0.5, 0.01,
            key=f"intensity_{f}"
        )

# ---- Sidebar: Editing Tools ----
st.sidebar.header("🛠 Editing Tools")
denoise_strength = st.sidebar.slider("Denoise Strength 🧹", 0.0, 3.0, 0.0, 0.01)
apply_denoise = st.sidebar.button("Apply Denoise 🧹")
rotate_90 = st.sidebar.checkbox("Rotate 90° 🔄")
apply_crop = st.sidebar.checkbox("✂ Crop")
apply_text = st.sidebar.checkbox("📝 Add Text")

# ---- File uploader ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    uploaded_image = Image.open(uploaded_file).convert("RGB")

    if ("base_image" not in st.session_state) or ("uploaded_file_name" not in st.session_state) or (st.session_state.uploaded_file_name != uploaded_file.name):
        st.session_state.base_image = uploaded_image.copy()
        st.session_state.edited_image = uploaded_image.copy()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.history = [uploaded_image.copy()]

    # start from base image every time
    img = st.session_state.base_image.copy()
    preview_img = img.copy()

    # ---- Crop ----
    if apply_crop:
        st.write("✂ Drag the box to crop the image")
        cropped_img = st_cropper(img, realtime_update=True, box_color="red", aspect_ratio=None)
        if st.button("Apply Crop"):
            img = cropped_img
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Crop applied!")

    # ---- Live Denoise Preview ----
    if denoise_strength > 0:
        cv_img = cv2.cvtColor(np.array(preview_img), cv2.COLOR_RGB2BGR)
        # قوة التنضيف على حسب السلايدر
        if denoise_strength <= 1:
            scaled_strength = int(1 + denoise_strength * 15)
        else:
            scaled_strength = int(15 + (denoise_strength - 1) * 20)
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, scaled_strength, scaled_strength, 7, 21)
        preview_img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

    # ---- Apply Denoise (commit to history) ----
    if apply_denoise and denoise_strength > 0:
        st.session_state.base_image = preview_img.copy()
        st.session_state.history.append(preview_img.copy())
        st.success("Denoise applied!")

    # ---- Rotation ----
    if rotate_90:
        if st.button("Apply 90° Rotation 🔄"):
            rotated = img.rotate(90, expand=True)
            st.session_state.base_image = rotated.copy()
            st.session_state.history.append(rotated.copy())
            st.success("Rotation applied!")

    # ---- Filters ----
    temp_img = preview_img.copy()
    if apply_filters:
        for f in apply_filters:
            intensity = filter_intensities.get(f, 1.0)
            if f == "Grayscale":
                temp_img = grayscale_filter(temp_img, intensity)
            elif f == "Sepia":
                temp_img = sepia_filter(temp_img, intensity)
            elif f == "Blur":
                temp_img = blur_filter(temp_img, intensity)
            elif f == "Cartoon":
                temp_img = cartoon_filter(temp_img, intensity)
            elif f == "Cartoon Colorful":
                temp_img = cartoon_colorful_filter(temp_img, intensity)
            elif f == "HDR Enhanced":
                temp_img = hdr_enhanced_filter(temp_img, intensity)

    # ---- Adjustments ----
    temp_img = ImageEnhance.Brightness(temp_img).enhance(brightness)
    temp_img = ImageEnhance.Contrast(temp_img).enhance(contrast)
    if sharpness > 2.0:
        temp_array = np.array(temp_img)
        temp_array = cv2.detailEnhance(temp_array, sigma_s=10 * (sharpness - 2.0) / 3.0, sigma_r=0.15)
        temp_img = Image.fromarray(temp_array)
    temp_img = ImageEnhance.Sharpness(temp_img).enhance(sharpness)

    st.session_state.edited_image = temp_img

    final_width, final_height = get_mobile_dimensions(img)
    st.image(st.session_state.edited_image, caption="Edited Image", use_column_width=False, width=final_width)

    # ---- Undo Button ----
    if st.button("↩ Undo"):
        if len(st.session_state.history) > 1:
            st.session_state.history.pop()
            st.session_state.base_image = st.session_state.history[-1].copy()
            st.session_state.edited_image = st.session_state.base_image.copy()
            st.success("Undo applied!")
        else:
            st.warning("No more steps to undo!")

    buf = io.BytesIO()
    st.session_state.edited_image.save(buf, format="PNG")
    st.download_button("💾 Download Edited Image", data=buf.getvalue(),
                       file_name="edited_image.png", mime="image/png")