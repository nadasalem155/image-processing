import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import io
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Image Editing App", layout="centered")

st.title("📸 Image Editing App")

# Sidebar options
st.sidebar.header("Editing Tools")
add_noise = st.sidebar.checkbox("Add Noise")
apply_filters = st.sidebar.multiselect(
    "Filters",
    ["BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "SHARPEN", "SMOOTH", "EMBOSS", "FIND_EDGES", "Solarize", "Posterize"]
)
rotate_90 = st.sidebar.checkbox("Rotate 90°")
apply_crop = st.sidebar.checkbox("Crop with Mouse")
apply_remove = st.sidebar.checkbox("Remove with Brush")
apply_text = st.sidebar.checkbox("Add Text")
apply_emoji = st.sidebar.checkbox("Add Emoji")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Keep track of edits
    if "edited_image" not in st.session_state:
        st.session_state.edited_image = image.copy()

    img = st.session_state.edited_image.copy()
    st.image(img, caption="Current Image")  # no stretching now ✅

    # Crop with mouse
    if apply_crop:
        st.write("✂ Drag the box to crop the image")
        cropped_img = st_cropper(img, realtime_update=True, box_color="red", aspect_ratio=None)
        if st.sidebar.button("Apply Crop"):
            img = cropped_img

    # Noise
    if add_noise and st.sidebar.button("Apply Noise"):
        img_arr = np.array(img)
        noise = np.random.randint(0, 50, img_arr.shape, dtype="uint8")
        img_arr = cv2.add(img_arr, noise)
        img = Image.fromarray(img_arr)

    # Filters
    for f in apply_filters:
        if f == "BLUR":
            img = img.filter(ImageFilter.BLUR)
        elif f == "CONTOUR":
            img = img.filter(ImageFilter.CONTOUR)
        elif f == "DETAIL":
            img = img.filter(ImageFilter.DETAIL)
        elif f == "EDGE_ENHANCE":
            img = img.filter(ImageFilter.EDGE_ENHANCE)
        elif f == "SHARPEN":
            img = img.filter(ImageFilter.SHARPEN)
        elif f == "SMOOTH":
            img = img.filter(ImageFilter.SMOOTH)
        elif f == "EMBOSS":
            img = img.filter(ImageFilter.EMBOSS)
        elif f == "FIND_EDGES":
            img = img.filter(ImageFilter.FIND_EDGES)
        elif f == "Solarize":
            img = ImageOps.solarize(img, threshold=128)
        elif f == "Posterize":
            img = ImageOps.posterize(img, bits=4)

    # Rotate
    if rotate_90 and st.sidebar.button("Apply 90° Rotation"):
        img = img.rotate(90, expand=True)

    # Remove (simple demo with mask)
    if apply_remove and st.sidebar.button("Apply Remove"):
        cv_img = np.array(img)
        mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)
        mask[50:150, 50:150] = 255
        inpainted = cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)
        img = Image.fromarray(inpainted)

    # Text
    if apply_text and st.sidebar.button("Apply Text"):
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((50, 50), "Hello!", fill=(255, 0, 0), font=font)

    # Emoji
    if apply_emoji and st.sidebar.button("Apply Emoji"):
        draw = ImageDraw.Draw(img)
        draw.text((100, 100), "😊", fill=(0, 0, 0))

    # Save edits
    st.session_state.edited_image = img.copy()

    st.image(img, caption="Edited Image")  # final image

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.download_button("Download Edited Image", data=buf.getvalue(),
                       file_name="edited_image.png", mime="image/png")

else:
    st.warning("⚠ Please upload an image to start editing.")