import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2
import io
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

# ---- Filter Functions ----
def cartoon_filter(img, intensity=1.0):
    if intensity == 0:
        return img
    img_array = np.array(img)
    
    # Apply pyrMeanShiftFiltering for color segmentation (cartoonish flat areas)
    shifted = cv2.pyrMeanShiftFiltering(img_array, sp=15, sr=30)  # sp and sr control the level of segmentation
    
    # Get edges using adaptive thresholding (black lines on white areas)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)  # White areas, black lines
    
    # Combine: apply colors where mask is white, black lines where mask is black
    cartoon = cv2.bitwise_and(shifted, shifted, mask=edges)
    
    # Blend with original based on intensity
    result = cv2.addWeighted(img_array, 1 - intensity, cartoon, intensity, 0)
    
    return Image.fromarray(result)

def cartoon_colorful_filter(img, intensity=1.0):
    img_array = np.array(img)
    
    # Lighter bilateral filter for natural smoothing
    color = cv2.bilateralFilter(img_array, d=7, sigmaColor=150, sigmaSpace=150)
    
    # Mild HSV adjustment for colorful but natural boost
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, int(30 * intensity))  # Milder saturation boost
    s = np.clip(s, 0, 255)
    v = cv2.add(v, int(20 * intensity))  # Milder brightness boost
    v = np.clip(v, 0, 255)
    hsv = cv2.merge([h, s, v])
    colorful = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Softer edges with adaptive thresholding
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)  # Smaller blur for natural details
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)  # Natural, smooth edges
    edges = cv2.GaussianBlur(edges, (3, 3), 0)  # Light smoothing
    # No invert needed: adaptive gives white areas, black lines
    
    # Combine with mask
    colorful = cv2.bitwise_and(colorful, colorful, mask=edges)
    
    # Subtle detail enhancement
    colorful = cv2.detailEnhance(colorful, sigma_s=5 * intensity, sigma_r=0.1 * intensity)
    
    # Balanced blend with original for natural look
    result = cv2.addWeighted(img_array, 0.6 + 0.4 * (1 - intensity), colorful, 0.4 * intensity, 0)
    
    return Image.fromarray(result)

def blur_filter(img, intensity=1.0):
    img_array = np.array(img)
    # Adjust kernel size based on intensity (min 5, max 15)
    kernel_size = int(5 + 10 * intensity)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # Ensure odd kernel size
    blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    # Blend with original image based on intensity
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
    # Blend with original image based on intensity
    result = cv2.addWeighted(img_array, 1 - intensity, enhanced, intensity, 0)
    return Image.fromarray(result)

def grayscale_filter(img, intensity=1.0):
    img_array = np.array(img)
    cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # Blend with original image based on intensity
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
    # Blend with original image based on intensity
    result = cv2.addWeighted(img_array, 1 - intensity, sepia_rgb, intensity, 0)
    return Image.fromarray(result)

# ---- Page config ----
st.set_page_config(page_title="📸🎨🖌 Image Editing App", layout="centered")
st.title("📸🎨🖌 Image Editing App – Easy & Fun Photo Editing")

# ---- Function to resize image for mobile ----
def get_mobile_dimensions(pil_img, max_width=350):
    aspect_ratio = pil_img.height / pil_img.width
    width = min(pil_img.width, max_width)
    height = int(width * aspect_ratio)
    return width, height

# ---- Sidebar: Adjustments ----
st.sidebar.header("⚙ Adjustments")
brightness = st.sidebar.slider("Brightness ☀", 0.0, 2.0, 1.0, 0.01)
contrast = st.sidebar.slider("Contrast 🎚", 0.0, 2.0, 1.0, 0.01)
sharpness = st.sidebar.slider("Sharpness 🔪", 0.0, 5.0, 1.0, 0.01)  # Expanded range for stronger sharpness

# ---- Sidebar: Filters & Effects ----
st.sidebar.header("🎨 Filters & Effects")
filter_options = ["Grayscale", "Sepia", "Blur", "Cartoon", "Cartoon Colorful", "HDR Enhanced"]
apply_filters = st.sidebar.multiselect("Filters 🎭", filter_options)

# Dictionary to store filter intensities
filter_intensities = {}
for f in filter_options:
    if f in apply_filters:
        filter_intensities[f] = st.sidebar.slider(f"Intensity of {f} (%)", 0.0, 1.0, 1.0, 0.01, key=f"intensity_{f}")

# ---- Sidebar: Editing Tools ----
st.sidebar.header("🛠 Editing Tools")
denoise = st.sidebar.checkbox("Denoise 🧹")
rotate_90 = st.sidebar.checkbox("Rotate 90° 🔄")
apply_crop = st.sidebar.checkbox("✂ Crop")
apply_remove = st.sidebar.checkbox("🖌 Remove")
apply_text = st.sidebar.checkbox("📝 Add Text")

# ---- File uploader ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    uploaded_image = Image.open(uploaded_file).convert("RGB")
    
    # ---- Reset session state for new image ----
    if ("base_image" not in st.session_state) or ("uploaded_file_name" not in st.session_state) or (st.session_state.uploaded_file_name != uploaded_file.name):
        st.session_state.base_image = uploaded_image.copy()
        st.session_state.edited_image = uploaded_image.copy()
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.history = [uploaded_image.copy()]

    img = st.session_state.base_image.copy()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_png = Image.open(buf)
    final_width, final_height = get_mobile_dimensions(img)

    # ---- Crop Tool ----
    if apply_crop:
        st.write("✂ Drag the box to crop the image")
        cropped_img = st_cropper(img_png, realtime_update=True, box_color="red", aspect_ratio=None)
        if st.button("Apply Crop"):
            img = cropped_img
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Crop applied!")

    # ---- Remove Tool ----
    if apply_remove:
        st.write("🖌 Draw over the area you want to remove")
        canvas_width, canvas_height = get_mobile_dimensions(img_png)
        canvas_result = st_canvas(
            fill_color="rgba(255,255,255,0)",
            stroke_width=20,
            stroke_color="white",
            background_image=img_png,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key="remove_canvas",
        )
        if st.button("Apply Remove"):
            if canvas_result.image_data is not None:
                mask = np.array(canvas_result.image_data)[:, :, 3]
                mask = cv2.resize(mask, (img.width, img.height), interpolation=cv2.INTER_NEAREST)
                cv_img = np.array(img)
                inpainted = cv2.inpaint(cv_img, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
                img = Image.fromarray(inpainted)
                st.session_state.base_image = img.copy()
                st.session_state.history.append(img.copy())
                st.success("Object removed!")

    # ---- Denoise ----
    if denoise:
        if st.button("Apply Denoise 🧹"):
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # Check if the image has noise by using standard deviation
            if np.std(cv_img) < 1:
                st.warning("No noise detected in the image!")
            else:
                denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)
                img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
                st.session_state.base_image = img.copy()
                st.session_state.history.append(img.copy())
                st.success("Noise removed!")

    # ---- Rotate ----
    if rotate_90:
        if st.button("Apply 90° Rotation 🔄"):
            img = img.rotate(90, expand=True)
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Rotation applied!")

    # ---- Live Filter Preview ----
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

        # Apply adjustments (Brightness, Contrast, Sharpness) to the preview
        temp_img = ImageEnhance.Brightness(temp_img).enhance(brightness)
        temp_img = ImageEnhance.Contrast(temp_img).enhance(contrast)
        # Enhanced sharpness with detailEnhance for high values
        if sharpness > 2.0:
            temp_array = np.array(temp_img)
            temp_array = cv2.detailEnhance(temp_array, sigma_s=10 * (sharpness - 2.0) / 3.0, sigma_r=0.15)
            temp_img = Image.fromarray(temp_array)
        temp_img = ImageEnhance.Sharpness(temp_img).enhance(sharpness)

        st.image(temp_img, caption="Filter Preview", use_column_width=False, width=final_width)

        if st.button("Apply Filters 🎭"):
            img = temp_img.copy()
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Filters applied!")

    # ---- Text ----
    if apply_text:
        st.write("📝 Add Text (choose size & color above the image)")
        text_input = st.text_input("Enter your text", "Hello!")
        text_size = st.slider("Text Size 🔠", 50, 500, 100)
        text_color = st.color_picker("Text Color 🎨", "#FF0000")
        box_data = st_cropper(img_png, realtime_update=True, box_color="blue", aspect_ratio=None, return_type="box")
        if st.button("Apply Text"):
            draw = ImageDraw.Draw(img)
            scaled_size = int(text_size * 1.5)
            try:
                font = ImageFont.truetype("arial.ttf", scaled_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", scaled_size)
                except:
                    font = ImageFont.load_default()
                    st.warning("No TrueType font found. Text size may be limited. Please place 'arial.ttf' or another .ttf font in the app directory.")
            left = box_data['left']
            top = box_data['top']
            draw.text((left, top), text_input, fill=text_color, font=font)
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Text applied!")

    # ---- Apply sliders permanently ----
    temp_img = st.session_state.base_image.copy()
    temp_img = ImageEnhance.Brightness(temp_img).enhance(brightness)
    temp_img = ImageEnhance.Contrast(temp_img).enhance(contrast)
    # Enhanced sharpness with detailEnhance for high values
    if sharpness > 2.0:
        temp_array = np.array(temp_img)
        temp_array = cv2.detailEnhance(temp_array, sigma_s=10 * (sharpness - 2.0) / 3.0, sigma_r=0.15)
        temp_img = Image.fromarray(temp_array)
    temp_img = ImageEnhance.Sharpness(temp_img).enhance(sharpness)
    st.session_state.edited_image = temp_img

    # ---- Show final edited image ----
    st.image(st.session_state.edited_image, caption="Edited Image", use_column_width=False, width=final_width)

    # ---- Undo button ----
    if st.button("↩ Undo"):
        if len(st.session_state.history) > 1:
            st.session_state.history.pop()
            st.session_state.base_image = st.session_state.history[-1].copy()
            st.session_state.edited_image = st.session_state.base_image.copy()
            st.success("Undo applied!")
        else:
            st.warning("No more steps to undo!")

    # ---- Download button ----
    buf = io.BytesIO()
    st.session_state.edited_image.save(buf, format="PNG")
    st.download_button("💾 Download Edited Image", data=buf.getvalue(),
                       file_name="edited_image.png", mime="image/png")