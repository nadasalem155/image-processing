import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import io
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Image Editing App", layout="centered")
st.title("📸 Image Editing App")

# Sidebar options
st.sidebar.header("Editing Tools")
add_noise = st.sidebar.checkbox("Add Noise")
apply_filters = st.sidebar.multiselect(
    "Filters",
    ["BLUR", "CONTOUR", "DETAIL", "EDGE_ENHANCE", "SHARPEN", "SMOOTH", "EMBOSS",
     "FIND_EDGES", "Solarize", "Posterize"]
)
rotate_90 = st.sidebar.checkbox("Rotate 90°")
apply_crop = st.sidebar.checkbox("Crop with Mouse")
apply_remove = st.sidebar.checkbox("Remove with Brush")
apply_text = st.sidebar.checkbox("Add Text")
apply_emoji = st.sidebar.checkbox("Add Emoji")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if "edited_image" not in st.session_state:
        st.session_state.edited_image = image.copy()

    img = st.session_state.edited_image.copy()

    # ---- Crop Tool ----
    if apply_crop:
        st.write("✂ Drag the box to crop the image")
        cropped_img = st_cropper(img, realtime_update=True, box_color="red", aspect_ratio=None)
        if st.sidebar.button("Apply Crop"):
            img = cropped_img
            st.session_state.edited_image = img.copy()

    # ---- Remove Tool ----
    if apply_remove:
        st.write("🖌 Draw over the area you want to remove")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="white",
            background_image=img,
            update_streamlit=True,
            height=img.height,
            width=img.width,
            drawing_mode="freedraw",
            key="remove_canvas",
        )
        if st.sidebar.button("Apply Remove") and canvas_result.image_data is not None:
            mask = np.array(canvas_result.image_data)[:, :, 3]  # alpha channel as mask
            cv_img = np.array(img)
            inpainted = cv2.inpaint(cv_img, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
            img = Image.fromarray(inpainted)
            st.session_state.edited_image = img.copy()

    # ---- Noise ----
    if add_noise and st.sidebar.button("Apply Noise"):
        img_arr = np.array(img)
        noise = np.random.randint(0, 50, img_arr.shape, dtype="uint8")
        img_arr = cv2.add(img_arr, noise)
        img = Image.fromarray(img_arr)
        st.session_state.edited_image = img.copy()

    # ---- Filters ----
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
        st.session_state.edited_image = img.copy()

    # ---- Rotate ----
    if rotate_90 and st.sidebar.button("Apply 90° Rotation"):
        img = img.rotate(90, expand=True)
        st.session_state.edited_image = img.copy()

    # ---- Text Tool ----
    if apply_text:
        text_input = st.text_input("Enter your text", "Hello!")
        font_size = st.sidebar.slider("Text Size", 10, 150, 40)
        st.write("📦 Drag the box to position your text")
        text_box = st_cropper(img, realtime_update=True, box_color="blue", aspect_ratio=None)
        if st.sidebar.button("Apply Text"):
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", font_size)
            box_w, box_h = text_box.size
            left = (img.width - box_w) // 2
            top = (img.height - box_h) // 2
            w, h = font.getsize(text_input)
            x = left + (box_w - w) // 2
            y = top + (box_h - h) // 2
            draw.text((x, y), text_input, fill=(255, 0, 0), font=font)
            st.session_state.edited_image = img.copy()

    # ---- Emoji Tool ----
    if apply_emoji:
        emoji_input = st.text_input("Enter Emoji", "😊")
        font_size = st.sidebar.slider("Emoji Size", 20, 200, 80)
        st.write("📦 Drag the box to position your emoji")
        emoji_box = st_cropper(img, realtime_update=True, box_color="green", aspect_ratio=None)
        if st.sidebar.button("Apply Emoji"):
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", font_size)
            box_w, box_h = emoji_box.size
            left = (img.width - box_w) // 2
            top = (img.height - box_h) // 2
            w, h = font.getsize(emoji_input)
            x = left + (box_w - w) // 2
            y = top + (box_h - h) // 2
            draw.text((x, y), emoji_input, fill=(0, 0, 0), font=font)
            st.session_state.edited_image = img.copy()

    # ---- Show final edited image ----
    st.image(st.session_state.edited_image, caption="Edited Image")

    # ---- Download ----
    buf = io.BytesIO()
    st.session_state.edited_image.save(buf, format="PNG")
    st.download_button("Download Edited Image", data=buf.getvalue(),
                       file_name="edited_image.png", mime="image/png")

else:
    st.warning("⚠ Please upload an image to start editing.")