import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import cv2
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Image Editor", layout="wide")

# Initialize session state
if "base_image" not in st.session_state:
    st.session_state.base_image = None
if "history" not in st.session_state:
    st.session_state.history = []

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    if st.session_state.base_image is None:
        st.session_state.base_image = img.copy()
        st.session_state.history = [img.copy()]
else:
    st.warning("Please upload an image to start.")

# Sidebar options
st.sidebar.header("Options")
apply_filter = st.sidebar.checkbox("Apply Filters")
apply_text = st.sidebar.checkbox("Add Text")
remove_object = st.sidebar.checkbox("Remove Object")
reset_image = st.sidebar.button("Reset Image")

if reset_image:
    if uploaded_file:
        st.session_state.base_image = Image.open(uploaded_file).convert("RGB").copy()
        st.session_state.history = [st.session_state.base_image.copy()]

if st.session_state.base_image:
    img = st.session_state.base_image.copy()
    img_np = np.array(img)
    img_png = img.copy()

    # ---- Filters ----
    if apply_filter:
        st.subheader("🎨 Filters")
        filter_type = st.radio(
            "Choose a filter:",
            ["None", "Blur", "Contour", "Detail", "Sharpen", "Emboss", "Cartoon"],
        )

        if filter_type == "Blur":
            img = img.filter(ImageFilter.GaussianBlur(5))
        elif filter_type == "Contour":
            img = img.filter(ImageFilter.CONTOUR)
        elif filter_type == "Detail":
            img = img.filter(ImageFilter.DETAIL)
        elif filter_type == "Sharpen":
            img = img.filter(ImageFilter.SHARPEN)
        elif filter_type == "Emboss":
            img = img.filter(ImageFilter.EMBOSS)
        elif filter_type == "Cartoon":
            # Cartoon effect using OpenCV
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
            )
            color = cv2.bilateralFilter(img_cv, 9, 250, 250)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            img = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

        st.session_state.base_image = img.copy()
        st.session_state.history.append(img.copy())

    # ---- Text ----
    if apply_text:
        st.write("📝 Add Text (choose size & color above the image)")
        text_input = st.text_input("Enter your text", "Hello!")
        text_size = st.slider("Text Size 🔠", 10, 300, 100)  # الافتراضي 100
        text_color = st.color_picker("Text Color 🎨", "#FF0000")

        # مكان الكتابة باستخدام crop box
        box_data = st_cropper(img_png, realtime_update=True, box_color="blue", aspect_ratio=None, return_type="box")
        
        if st.button("Apply Text"):
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", text_size)
            except:
                font = ImageFont.truetype("DejaVuSans.ttf", text_size)  # fallback
            left = box_data['left']
            top = box_data['top']
            draw.text((left, top), text_input, fill=text_color, font=font)
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Text applied!")

    # ---- Object Removal ----
    if remove_object:
        st.subheader("🩹 Remove Object")
        box_data = st_cropper(img_png, realtime_update=True, box_color="red", aspect_ratio=None, return_type="box")

        if st.button("Remove Selected Area"):
            left = int(box_data['left'])
            top = int(box_data['top'])
            width = int(box_data['width'])
            height = int(box_data['height'])
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            mask = np.zeros(img_cv.shape[:2], np.uint8)
            mask[top:top+height, left:left+width] = 255
            result = cv2.inpaint(img_cv, mask, 3, cv2.INPAINT_TELEA)
            img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Object removed!")

    # ---- Show Image ----
    st.image(img, caption="Current Image", use_container_width=True)

    # ---- Download ----
    st.download_button(
        "Download Edited Image",
        data=img.convert("RGB").tobytes(),
        file_name="edited_image.jpg",
        mime="image/jpeg",
    )