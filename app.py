import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_cropper import st_cropper

# Initialize session state
if "base_image" not in st.session_state:
    st.session_state.base_image = None
if "history" not in st.session_state:
    st.session_state.history = []
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

# Sidebar menu
st.sidebar.title("🖼️ Image Editing Options")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.session_state.base_image = Image.open(uploaded_file).convert("RGB")
    st.session_state.history = [st.session_state.base_image.copy()]
    st.session_state.image_uploaded = True

if st.session_state.base_image is not None:
    img = st.session_state.base_image.copy()
    img_array = np.array(img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_png = Image.fromarray(img_array)

    # Options
    apply_filter = st.sidebar.selectbox("Choose Filter 🎨", ["None", "Cartoon", "Gray", "Sepia", "Invert", "Blur"])
    apply_text = st.sidebar.checkbox("Add Text 📝")
    remove_part = st.sidebar.checkbox("Remove Part ❌")

    # ---- Apply Filters ----
    if apply_filter == "Gray":
        img = Image.fromarray(img_gray)
    elif apply_filter == "Sepia":
        sepia = cv2.transform(img_array, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        img = Image.fromarray(sepia)
    elif apply_filter == "Invert":
        invert = cv2.bitwise_not(img_array)
        img = Image.fromarray(invert)
    elif apply_filter == "Blur":
        blur = cv2.GaussianBlur(img_array, (15, 15), 0)
        img = Image.fromarray(blur)
    elif apply_filter == "Cartoon":
        gray = cv2.medianBlur(img_gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_array, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        img = Image.fromarray(cartoon)

    # ---- Text ----
    if apply_text:
        st.write("📝 Add Text (choose size & color above the image)")
        text_input = st.text_input("Enter your text", "Hello!")
        text_size = st.slider("Text Size 🔠", 10, 300, 100)  # الافتراضي أكبر
        text_color = st.color_picker("Text Color 🎨", "#FF0000")
        box_data = st_cropper(img_png, realtime_update=True, box_color="blue", aspect_ratio=None, return_type="box")
        if st.button("Apply Text"):
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", text_size)
            except:
                font = ImageFont.truetype("DejaVuSans.ttf", text_size)  # fallback كبير وواضح
            left = box_data['left']
            top = box_data['top']
            draw.text((left, top), text_input, fill=text_color, font=font)
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Text applied!")

    # ---- Remove Part ----
    if remove_part:
        st.write("❌ Select part to remove")
        box_data = st_cropper(img_png, realtime_update=True, box_color="red", aspect_ratio=None, return_type="box")
        if st.button("Remove Selected Area"):
            left, top, width, height = box_data['left'], box_data['top'], box_data['width'], box_data['height']
            img_array[top:top+height, left:left+width] = 255  # fill with white
            img = Image.fromarray(img_array)
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Part removed!")

    # Display final image
    st.image(img, caption="Current Image", use_container_width=True)

    # Undo button
    if st.button("Undo ↩️"):
        if len(st.session_state.history) > 1:
            st.session_state.history.pop()
            st.session_state.base_image = st.session_state.history[-1].copy()
            st.success("Undo successful!")

else:
    st.write("⬅️ Please upload an image to start editing.")