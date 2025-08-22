import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import numpy as np
import cv2
from streamlit_cropper import st_cropper

st.set_page_config(page_title="🖼️ Image Editor", layout="wide")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "base_image" not in st.session_state:
    st.session_state.base_image = None

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    if st.session_state.base_image is None:
        st.session_state.base_image = img.copy()
    img = st.session_state.base_image.copy()
    img_png = img.copy()

    st.image(img, caption="Current Image", use_container_width=True)

    # Sidebar options
    st.sidebar.header("Tools")
    apply_filter = st.sidebar.selectbox("🎨 Filters", ["None", "Cartoon", "Sketch", "Blur", "Sharpen"])
    apply_text = st.sidebar.checkbox("📝 Add Text")
    remove_bg = st.sidebar.checkbox("❌ Remove Background")

    # ---- Filters ----
    if apply_filter != "None":
        img_np = np.array(img)
        if apply_filter == "Cartoon":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(img_np, 9, 250, 250)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            img = Image.fromarray(cartoon)

        elif apply_filter == "Sketch":
            gray, sketch = cv2.pencilSketch(img_np, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            img = Image.fromarray(sketch)

        elif apply_filter == "Blur":
            img = img.filter(ImageFilter.GaussianBlur(4))

        elif apply_filter == "Sharpen":
            img = img.filter(ImageFilter.SHARPEN)

        st.session_state.base_image = img.copy()
        st.session_state.history.append(img.copy())
        st.success(f"{apply_filter} filter applied!")

    # ---- Text ----
    if apply_text:
        st.write("📝 Add Text (choose size & color above the image)")
        text_input = st.text_input("Enter your text", "Hello!")
        text_size = st.slider("Text Size 🔠", 10, 300, 100)  # default 100
        text_color = st.color_picker("Text Color 🎨", "#FF0000")

        # Crop box to choose text position
        box_data = st_cropper(img_png, realtime_update=True,
                              box_color="blue", aspect_ratio=None, return_type="box")

        if st.button("Apply Text"):
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", text_size)
            except:
                font = ImageFont.truetype("DejaVuSans.ttf", text_size)  # fallback واضح
            left = box_data['left']
            top = box_data['top']
            draw.text((left, top), text_input, fill=text_color, font=font)
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Text applied!")

    # ---- Background Removal ----
    if remove_bg:
        import rembg
        img_no_bg = rembg.remove(np.array(img))
        img = Image.fromarray(img_no_bg)
        st.session_state.base_image = img.copy()
        st.session_state.history.append(img.copy())
        st.success("Background removed!")

    # ---- Undo ----
    if st.sidebar.button("Undo ↩️") and st.session_state.history:
        st.session_state.history.pop()
        if st.session_state.history:
            st.session_state.base_image = st.session_state.history[-1].copy()
            st.image(st.session_state.base_image, caption="Undo Applied", use_container_width=True)
        else:
            st.session_state.base_image = None
            st.warning("No more steps to undo.")

    # ---- Download ----
    st.sidebar.download_button("💾 Download Image",
                               data=img.tobytes(),
                               file_name="edited.png",
                               mime="image/png")