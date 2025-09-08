import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2
import io
from streamlit_cropper import st_cropper

# ---- Filter Functions ----
def cartoon_filter(img, intensity=1.0):
    if intensity == 0:
        return img
    img_array = np.array(img)

    def color_quantization(im, k):
        data = np.float32(im).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
        ret, label, center = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        result = center[label.flatten()]
        result = result.reshape(im.shape)
        return result

    scale = 0.5
    small = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    k = max(4, 16 - int(12 * intensity))
    quantized_small = color_quantization(small, k)
    quantized = cv2.resize(quantized_small, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    block_size = 9
    c = 2 + int(3 * intensity)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)
    result = cv2.addWeighted(img_array, 1 - intensity, cartoon, intensity, 0)
    return Image.fromarray(result)

def cartoon_colorful_filter(img, intensity=1.0):
    img_array = np.array(img)
    color = cv2.bilateralFilter(img_array, 9, 300, 300)
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, int(50 * intensity))
    s = np.clip(s, 0, 255)
    v = cv2.add(v, int(30 * intensity))
    v = np.clip(v, 0, 255)
    hsv = cv2.merge([h, s, v])
    colorful = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=int(3 * intensity))
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edges = cv2.bitwise_not(edges)
    colorful = cv2.bitwise_and(colorful, colorful, mask=edges)
    colorful = cv2.detailEnhance(colorful, sigma_s=10 * intensity, sigma_r=0.15 * intensity)
    result = cv2.addWeighted(img_array, 1 - intensity, colorful, intensity, 0)
    return Image.fromarray(result)

def blur_filter(img, intensity=1.0):
    img_array = np.array(img)
    kernel_size = int(5 + 10 * intensity)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    result = cv2.addWeighted(img_array, 1 - intensity, blurred, intensity, 0)
    return Image.fromarray(result)

def hdr_enhanced_filter(img, intensity=1.0):
    img_array = np.array(img)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0 * intensity, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    result = cv2.addWeighted(img_array, 1 - intensity, enhanced, intensity, 0)
    return Image.fromarray(result)

def grayscale_filter(img, intensity=1.0):
    img_array = np.array(img)
    cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    result = cv2.addWeighted(img_array, 1 - intensity, gray_rgb, intensity, 0)
    return Image.fromarray(result)

def sepia_filter(img, intensity=1.0):
    img_array = np.array(img)
    cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]]) * intensity
    sepia = cv2.transform(cv_img, sepia_matrix)
    sepia = np.clip(sepia, 0, 255)
    sepia_rgb = cv2.cvtColor(sepia, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(img_array, 1 - intensity, sepia_rgb, intensity, 0)
    return Image.fromarray(result)

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
sharpness = st.sidebar.slider("Sharpness 🔪", -2.0, 3.0, 0.0, 0.01)
denoise_strength = st.sidebar.slider("Denoise Strength 🧹", 0, 30, 0, 1)

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

    # ✅ Apply denoise dynamically if slider moved
    if denoise_strength > 0:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, denoise_strength, denoise_strength, 7, 21)
        img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        st.session_state.base_image = img.copy()

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_png = Image.open(buf)
    final_width, final_height = get_mobile_dimensions(img)

    if apply_crop:
        st.write("✂ Drag the box to crop the image")
        cropped_img = st_cropper(img_png, realtime_update=True, box_color="red", aspect_ratio=None)
        if st.button("Apply Crop"):
            img = cropped_img
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Crop applied!")

    if rotate_90:
        if st.button("Apply 90° Rotation 🔄"):
            img = img.rotate(90, expand=True)
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Rotation applied!")

    temp_img = img.copy()
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

        if brightness != 0.0:
            temp_img = ImageEnhance.Brightness(temp_img).enhance(1.0 + brightness)
        if contrast != 0.0:
            temp_img = ImageEnhance.Contrast(temp_img).enhance(1.0 + contrast)
        if sharpness != 0.0:
            if sharpness > 2.0:
                temp_array = np.array(temp_img)
                temp_array = cv2.detailEnhance(temp_array, sigma_s=10 * ((sharpness - 2.0) / 3.0), sigma_r=0.15)
                temp_img = Image.fromarray(temp_array)
            temp_img = ImageEnhance.Sharpness(temp_img).enhance(1.0 + sharpness)

        st.image(temp_img, caption="Filter Preview", use_column_width=False, width=final_width)

        if st.button("Apply Filters 🎭"):
            img = temp_img.copy()
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Filters applied!")

    temp_img = st.session_state.base_image.copy()
    if brightness != 0.0:
        temp_img = ImageEnhance.Brightness(temp_img).enhance(1.0 + brightness)
    if contrast != 0.0:
        temp_img = ImageEnhance.Contrast(temp_img).enhance(1.0 + contrast)
    if sharpness != 0.0:
        if sharpness > 2.0:
            temp_array = np.array(temp_img)
            temp_array = cv2.detailEnhance(temp_array, sigma_s=10 * ((sharpness - 2.0) / 3.0), sigma_r=0.15)
            temp_img = Image.fromarray(temp_array)
        temp_img = ImageEnhance.Sharpness(temp_img).enhance(1.0 + sharpness)

    st.session_state.edited_image = temp_img
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