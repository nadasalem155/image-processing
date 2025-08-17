# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError as e:
    st.error(f"Failed to import streamlit-drawable-canvas: {e}. Ensure it is installed and compatible with Streamlit.")
    st.stop()
import io

st.set_page_config(page_title="Mini Photoshop 🎨", layout="wide")

# Session state
if "orig_image" not in st.session_state:
    st.session_state.orig_image = None
if "working_image" not in st.session_state:
    st.session_state.working_image = None
if "history" not in st.session_state:
    st.session_state.history = []

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.session_state.orig_image = img
    st.session_state.working_image = img.copy()
    st.session_state.history = [img.copy()]

# Helper functions
def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def try_load_font(size=24):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()

# Sidebar tools
st.sidebar.header("🛠 Tools")
filter_choice = st.sidebar.selectbox("🎨 Choose Filter", [
    "None", "Grayscale", "Sepia", "Invert", "Blur", "Edge", "Cartoon", 
    "Emboss", "Sharpen", "Sketch", "HDR"
])
brightness = st.sidebar.slider("☀ Brightness", -100, 100, 0)
contrast = st.sidebar.slider("🌗 Contrast", 0.5, 3.0, 1.0)
sharpness = st.sidebar.slider("🔪 Sharpness", 0.5, 3.0, 1.0)
denoise = st.sidebar.slider("🧹 Noise Removal", 0, 30, 0)
rotate_choice = st.sidebar.radio("🔄 Rotate", ["None", "90°", "180°", "270°", "360° Reset"])
do_crop = st.sidebar.checkbox("✂ Crop Mode", key="crop_mode")
do_remove = st.sidebar.checkbox("🧽 Remove Mode")
add_text = st.sidebar.text_input("📝 Add Text")
text_size = st.sidebar.slider("📏 Text Size", 12, 100, 36)
text_color = st.sidebar.color_picker("🎨 Text Color", "#FFFF00")
add_emoji = st.sidebar.selectbox("😂 Emoji", ["None", "😂", "😍", "🔥", "👍", "❤", "😎"])
emoji_size = st.sidebar.slider("📏 Emoji Size", 12, 100, 36)

# Apply filters & enhancements
def apply_edits():
    if not st.session_state.orig_image:
        return
    cv_img = pil_to_cv2(st.session_state.working_image)

    # Filters
    try:
        if filter_choice == "Grayscale":
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        elif filter_choice == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
            cv_img = cv2.transform(cv_img, kernel)
            cv_img = np.clip(cv_img, 0, 255).astype(np.uint8)
        elif filter_choice == "Invert":
            cv_img = cv2.bitwise_not(cv_img)
        elif filter_choice == "Blur":
            cv_img = cv2.GaussianBlur(cv_img, (15, 15), 0)
        elif filter_choice == "Edge":
            edges = cv2.Canny(cv_img, 100, 200)
            cv_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif filter_choice == "Cartoon":
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(cv_img, 9, 250, 250)
            cv_img = cv2.bitwise_and(color, color, mask=edges)
        elif filter_choice == "Emboss":
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            cv_img = cv2.filter2D(cv_img, -1, kernel) + 128
        elif filter_choice == "Sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            cv_img = cv2.filter2D(cv_img, -1, kernel)
        elif filter_choice == "Sketch":
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            cv_img = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        elif filter_choice == "HDR":
            cv_img = cv2.detailEnhance(cv_img, sigma_s=12, sigma_r=0.15)
    except Exception as e:
        st.warning(f"Filter error: {e}")

    # Brightness & Contrast
    cv_img = cv2.convertScaleAbs(cv_img, alpha=contrast, beta=brightness)

    # Denoise
    if denoise > 0:
        try:
            cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, denoise, denoise, 7, 21)
        except Exception as e:
            st.warning(f"Denoise error: {e}")

    # Sharpness
    pil_img = cv2_to_pil(cv_img)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharpness)

    # Rotation
    if rotate_choice == "90°":
        pil_img = pil_img.rotate(-90, expand=True)
    elif rotate_choice == "180°":
        pil_img = pil_img.rotate(180, expand=True)
    elif rotate_choice == "270°":
        pil_img = pil_img.rotate(-270, expand=True)
    elif rotate_choice == "360° Reset":
        pil_img = st.session_state.orig_image.copy()

    st.session_state.working_image = pil_img
    st.session_state.history.append(pil_img.copy())

# Draw canvas
if st.session_state.working_image:
    apply_edits()

    img = st.session_state.working_image.copy()
    st.write("✏ Draw on Image")

    # Set drawing mode
    if do_crop:
        drawing_mode = "rect"
    elif do_remove:
        drawing_mode = "freedraw"
    elif add_text or add_emoji != "None":
        drawing_mode = "rect"
    else:
        drawing_mode = "transform"

    # Canvas with dynamic size
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=5,
        stroke_color="red",
        background_image=img,
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Crop apply
    if do_crop and st.button("Apply Crop"):
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][0]
            if obj["type"] == "rect":
                left, top, width, height = obj["left"], obj["top"], obj["width"], obj["height"]
                # Ensure valid coordinates
                left = max(0, int(left))
                top = max(0, int(top))
                right = min(img.width, int(left + width))
                bottom = min(img.height, int(top + height))
                if right > left and bottom > top:
                    st.session_state.working_image = st.session_state.working_image.crop((left, top, right, bottom))
                    st.session_state.history.append(st.session_state.working_image.copy())
                    st.success("Image cropped successfully!")
                    # Uncheck crop mode
                    st.session_state.crop_mode = False
                else:
                    st.error("Invalid crop coordinates. Draw a valid rectangle.")
            else:
                st.error("No rectangle drawn for cropping.")

    # Remove apply
    if do_remove and st.button("Apply Remove"):
        if canvas_result.image_data is not None:
            # Resize mask to match image dimensions
            mask = canvas_result.image_data[:, :, 3] > 0
            mask = mask.astype(np.uint8) * 255
            mask = cv2.resize(mask, (st.session_state.working_image.width, st.session_state.working_image.height), interpolation=cv2.INTER_NEAREST)
            cv_img = pil_to_cv2(st.session_state.working_image)
            try:
                inpainted = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
                st.session_state.working_image = cv2_to_pil(inpainted)
                st.session_state.history.append(st.session_state.working_image.copy())
                st.success("Object removed successfully!")
            except Exception as e:
                st.error(f"Inpainting error: {e}")
        else:
            st.error("No mask drawn for removal.")

    # Add text or emoji
    if (add_text or add_emoji != "None") and st.button("Apply Text/Emoji"):
        if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
            obj = canvas_result.json_data["objects"][0]
            if obj["type"] == "rect":
                left, top = int(obj["left"]), int(obj["top"])
                draw = ImageDraw.Draw(st.session_state.working_image)
                font = try_load_font(size=text_size if add_text else emoji_size)
                if add_text:
                    draw.text((left, top), add_text, fill=text_color, font=font)
                    st.success("Text added successfully!")
                if add_emoji != "None":
                    draw.text((left, top + (text_size if add_text else 0)), add_emoji, fill=text_color, font=font)
                    st.success("Emoji added successfully!")
                st.session_state.history.append(st.session_state.working_image.copy())
            else:
                st.error("No rectangle drawn for text/emoji placement.")
        else:
            st.error("No rectangle drawn for text/emoji placement.")

    # Undo button
    if st.button("Undo") and len(st.session_state.history) > 1:
        st.session_state.history.pop()
        st.session_state.working_image = st.session_state.history[-1].copy()

    # Show final
    st.image(st.session_state.working_image, use_column_width=False, width=350)

    # Save
    buf = io.BytesIO()
    st.session_state.working_image.save(buf, format="PNG")
    st.download_button("💾 Download Edited Image", data=buf.getvalue(), file_name="edited.png", mime="image/png")