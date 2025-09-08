import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import io

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

    # ---- زيادة تشبع الألوان ----
    hsv = cv2.cvtColor(cartoon, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, int(50 * intensity))
    s = np.clip(s, 0, 255)
    hsv = cv2.merge([h, s, v])
    colorful = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    result = cv2.addWeighted(img_array, 1 - intensity, colorful, intensity, 0)
    return Image.fromarray(result)


# ---- Streamlit App ----
st.title("Image Editor 🎨")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    filter_option = st.selectbox("Choose a filter", ["None", "Cartoon", "Cartoon Colorful"])
    intensity = st.slider("Intensity", 0.0, 1.0, 0.5, 0.01)

    if filter_option == "Cartoon":
        processed_img = cartoon_filter(image, intensity)
    elif filter_option == "Cartoon Colorful":
        processed_img = cartoon_colorful_filter(image, intensity)
    else:
        processed_img = image

    st.image(processed_img, caption="Filtered Image", use_column_width=True)

    buf = io.BytesIO()
    processed_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Image", data=byte_im, file_name="output.png", mime="image/png")