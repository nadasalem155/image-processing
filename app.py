import streamlit as st
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

st.set_page_config(page_title="Image Editor", layout="wide")

if "base_image" not in st.session_state:
    st.session_state.base_image = None
if "history" not in st.session_state:
    st.session_state.history = []

# Upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.session_state.base_image = Image.open(uploaded_file).convert("RGB")
    st.session_state.history = [st.session_state.base_image.copy()]

img = st.session_state.base_image

if img:
    st.image(img, caption="Current Image", use_container_width=True)

    # ======= Filters =======
    st.subheader("Filters")
    filter_choice = st.selectbox("Choose Filter", ["None", "Cartoon", "Oil Painting", "HDR"])
    if filter_choice != "None":
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if filter_choice == "Cartoon":
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(cv_img, d=9, sigmaColor=250, sigmaSpace=250)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            img = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

        elif filter_choice == "Oil Painting":
            oil = cv2.xphoto.oilPainting(cv_img, 7, 1)
            img = Image.fromarray(cv2.cvtColor(oil, cv2.COLOR_BGR2RGB))

        elif filter_choice == "HDR":
            hdr = cv2.detailEnhance(cv_img, sigma_s=12, sigma_r=0.15)
            img = Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))

        st.session_state.base_image = img.copy()
        st.session_state.history.append(img.copy())
        st.image(img, caption=f"Applied {filter_choice}", use_container_width=True)

    # ======= Remove Object =======
    st.subheader("Remove Object")
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="remove_canvas",
    )

    if st.button("Apply Remove"):
        if canvas_result.image_data is not None:
            mask = np.array(canvas_result.image_data)[:, :, 3]
            mask = cv2.resize(mask, (img.width, img.height), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8) * 255
            cv_img = np.array(img)
            inpainted = cv2.inpaint(cv_img, mask, 7, cv2.INPAINT_TELEA)
            img = Image.fromarray(inpainted)
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.success("Object removed!")
            st.image(img, caption="After Remove", use_container_width=True)

    # ======= Add Text =======
    st.subheader("Add Text")
    text_input = st.text_input("Enter Text")
    text_size = st.slider("Text Size (%)", 1, 20, 5)  # نسبة من العرض
    x = st.slider("X Position", 0, img.width, img.width // 4)
    y = st.slider("Y Position", 0, img.height, img.height // 4)

    if st.button("Apply Text"):
        if text_input:
            draw = ImageDraw.Draw(img)
            font_px = int((text_size / 100) * img.width)
            try:
                font = ImageFont.truetype("arial.ttf", font_px)
            except:
                font = ImageFont.load_default()
            draw.text((x, y), text_input, fill="white", font=font)
            st.session_state.base_image = img.copy()
            st.session_state.history.append(img.copy())
            st.image(img, caption="After Adding Text", use_container_width=True)

    # ======= Undo =======
    if st.button("Undo"):
        if len(st.session_state.history) > 1:
            st.session_state.history.pop()
            st.session_state.base_image = st.session_state.history[-1].copy()
            st.image(st.session_state.base_image, caption="Undo Applied", use_container_width=True)