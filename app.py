import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io
from streamlit_cropper import st_cropper

# ---- Filter Functions ----
def grayscale_filter(img, intensity=1.0):
    """Convert image to grayscale with intensity blend."""
    gray = ImageOps.grayscale(img).convert("RGB")
    return Image.blend(img, gray, intensity)

def sepia_filter(img, intensity=1.0):
    """Apply sepia effect with intensity."""
    img_array = np.array(img, dtype=np.float32)
    tr = 0.393 * img_array[:, :, 0] + 0.769 * img_array[:, :, 1] + 0.189 * img_array[:, :, 2]
    tg = 0.349 * img_array[:, :, 0] + 0.686 * img_array[:, :, 1] + 0.168 * img_array[:, :, 2]
    tb = 0.272 * img_array[:, :, 0] + 0.534 * img_array[:, :, 1] + 0.131 * img_array[:, :, 2]
    sepia = np.stack([tr, tg, tb], axis=2).clip(0, 255).astype(np.uint8)
    sepia_img = Image.fromarray(sepia)
    return Image.blend(img, sepia_img, intensity)

def blur_filter(img, intensity=0.5):
    """Gaussian blur with variable intensity."""
    radius = max(0, int(intensity * 10))
    return img.filter(ImageFilter.GaussianBlur(radius))

def cartoon_filter(img, intensity=0.5):
    """Cartoon effect using OpenCV stylization."""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cartoon = cv2.stylization(img_cv, sigma_s=150, sigma_r=0.25)
    cartoon_img = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
    return Image.blend(img, cartoon_img, intensity)

def cartoon_colorful_filter(img, intensity=0.5):
    """Colorful cartoon style with boosted saturation."""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cartoon = cv2.stylization(img_cv, sigma_s=150, sigma_r=0.25)

    hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.5, 0, 255)
    hsv = cv2.merge([h, s, v])
    colorful_cartoon = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

    colorful_img = Image.fromarray(cv2.cvtColor(colorful_cartoon, cv2.COLOR_BGR2RGB))
    return Image.blend(img, colorful_img, intensity)

def hdr_enhanced_filter(img, intensity=0.5):
    """HDR-like effect using detailEnhance."""
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hdr = cv2.detailEnhance(img_cv, sigma_s=12, sigma_r=0.15)
    hdr_img = Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
    return Image.blend(img, hdr_img, intensity)

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
brightness = st.sidebar.slider("Brightness ☀", -1.0, 1.0, 0.0, 0.01)
contrast = st.sidebar.slider("Contrast 🎚", -1.0, 1.0, 0.0, 0.01)
sharpness = st.sidebar.slider("Sharpness 🔪", -1.0, 2.0, 0.0, 0.01)

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
        if denoise_strength <= 2.5:
            scaled_strength = int(1 + denoise_strength * 15)
            denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, scaled_strength, scaled_strength, 7, 21)
        else:
            denoised = cv2.medianBlur(cv_img, 5)
        preview_img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

    # ---- Apply Denoise ----
    if apply_denoise and denoise_strength > 0:
        st.session_state.base_image = preview_img.copy()
        st.session_state.history.append(preview_img.copy())
        st.success("Denoise applied!")

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
    temp_img = ImageEnhance.Brightness(temp_img).enhance(1 + brightness)
    temp_img = ImageEnhance.Contrast(temp_img).enhance(1 + contrast)
    temp_img = ImageEnhance.Sharpness(temp_img).enhance(1 + sharpness)

    st.session_state.edited_image = temp_img
    final_width, final_height = get_mobile_dimensions(img)
    st.image(st.session_state.edited_image, caption="Edited Image", use_column_width=False, width=final_width)

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