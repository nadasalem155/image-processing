import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Image Processing App", layout="wide")

st.title("🖼️ Image Processing App")

# ---------------------- Upload Image ----------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Convert to RGB if not
    if img.mode != "RGB":
        img = img.convert("RGB")

    st.image(img, caption="Current Image", use_container_width=True)

    # Convert to numpy for OpenCV filters
    img_np = np.array(img)

    # ---------------------- Filters ----------------------
    st.subheader("🎨 Filters")
    filter_type = st.selectbox("Choose a filter", ["None", "Grayscale", "Sepia", "Cartoon", "Sketch", "Blur", "Sharpen"])

    if filter_type == "Grayscale":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    elif filter_type == "Sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        img_np = cv2.transform(img_np, sepia_filter)
        img_np = np.clip(img_np, 0, 255)

    elif filter_type == "Cartoon":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_np, 9, 250, 250)
        img_np = cv2.bitwise_and(color, color, mask=edges)

    elif filter_type == "Sketch":
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        img_np = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    elif filter_type == "Blur":
        img_np = cv2.GaussianBlur(img_np, (15, 15), 0)

    elif filter_type == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img_np = cv2.filter2D(img_np, -1, kernel)

    img_filtered = Image.fromarray(np.uint8(img_np))
    st.image(img_filtered, caption="After Filter", use_container_width=True)

    # ---------------------- Add Text ----------------------
    st.subheader("✍️ Add Text")
    user_text = st.text_input("Enter text to add")
    if user_text:
        img_text = img_filtered.copy()
        draw = ImageDraw.Draw(img_text)

        # Use truetype font for larger text size
        try:
            font = ImageFont.truetype("arial.ttf", 60)  # Large size
        except:
            font = ImageFont.load_default()

        draw.text((50, 50), user_text, fill="red", font=font)
        st.image(img_text, caption="With Text", use_container_width=True)

    # ---------------------- Remove Tool ----------------------
    st.subheader("🩹 Object Removal (Draw on area to remove)")
    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=30,
        stroke_color="white",
        background_image=img_filtered.convert("RGB"),
        background_color="white",  # Prevent black screen online
        update_streamlit=True,
        height=img_filtered.height,
        width=img_filtered.width,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        mask = np.array(canvas_result.image_data)[:, :, 3]
        mask = cv2.resize(mask, (img_filtered.width, img_filtered.height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8) * 255

        inpainted = cv2.inpaint(np.array(img_filtered), mask, 7, cv2.INPAINT_TELEA)
        img_inpainted = Image.fromarray(inpainted)
        st.image(img_inpainted, caption="After Removal", use_container_width=True)

else:
    st.warning("⚠️ Please upload an image to start")