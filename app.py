import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2
import io
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

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
sharpness = st.sidebar.slider("Sharpness 🔪", 0.0, 2.0, 1.0, 0.01)

# ---- Sidebar: Filters & Effects ----
st.sidebar.header("🎨 Filters & Effects")
filter_options = ["Grayscale", "Sepia", "Invert", "Blur", "Edge",
                  "Cartoon", "Emboss", "Sharpen", "Sketch", "HDR",
                  "Vintage", "Oil Painting", "Emboss Strong", "Cartoon Colorful", "HDR Enhanced", "Pencil Sketch Color"]
apply_filters = st.sidebar.multiselect("Filters 🎭", filter_options)

# ---- Sidebar: Editing Tools ----
st.sidebar.header("🛠 Editing Tools")
denoise = st.sidebar.checkbox("Denoise 🧹")
rotate_90 = st.sidebar.checkbox("Rotate 90° 🔄")
apply_crop = st.sidebar.checkbox("✂ Crop")
apply_remove = st.sidebar.checkbox("🖌 Remove")
apply_text = st.sidebar.checkbox("📝 Add Text")

# ---- File uploader ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

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
                denoised = cv2.medianBlur(cv_img, 5)
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
            if f == "Grayscale":
                temp_img = ImageOps.grayscale(temp_img).convert("RGB")
            elif f == "Sepia":
                arr = np.array(temp_img, dtype=np.float32)
                r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                tr = 0.393 * r + 0.769 * g + 0.189 * b
                tg = 0.349 * r + 0.686 * g + 0.168 * b
                tb = 0.272 * r + 0.534 * g + 0.131 * b
                arr[:, :, 0] = np.clip(tr, 0, 255)
                arr[:, :, 1] = np.clip(tg, 0, 255)
                arr[:, :, 2] = np.clip(tb, 0, 255)
                temp_img = Image.fromarray(arr.astype(np.uint8))
            elif f == "Invert":
                temp_img = ImageOps.invert(temp_img)
            elif f == "Blur":
                temp_img = temp_img.filter(ImageFilter.GaussianBlur(5))
            elif f == "Edge":
                temp_img = temp_img.filter(ImageFilter.FIND_EDGES)
            elif f == "Cartoon":
                cv_img2 = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.Canny(gray, 100, 200)  # Stronger edge detection
                edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)  # Thicken edges
                edges = cv2.bitwise_not(edges)  # Invert for black outlines
                color = cv2.bilateralFilter(cv_img2, 9, 300, 300)  # Strong smoothing
                cartoon = cv2.bitwise_and(color, color, mask=edges)
                temp_img = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
            elif f == "Emboss":
                temp_img = temp_img.filter(ImageFilter.EMBOSS)
            elif f == "Sharpen":
                temp_img = temp_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            elif f == "Sketch":
                cv_img2 = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                inv = cv2.bitwise_not(gray)
                blur = cv2.GaussianBlur(inv, (31, 31), 0)  # Stronger blur for smoother sketch
                inv_blur = cv2.bitwise_not(blur)
                sketch = cv2.divide(gray, inv_blur, scale=256.0)
                sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
                temp_img = Image.fromarray(sketch)
            elif f == "HDR":
                cv_img2 = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2BGR)
                hdr = cv2.detailEnhance(cv_img2, sigma_s=20, sigma_r=0.25)  # Stronger HDR effect
                temp_img = Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
                temp_img = ImageEnhance.Contrast(temp_img).enhance(1.4)  # Boost contrast
            elif f == "Vintage":
                arr = np.array(temp_img, dtype=np.float32)
                arr = arr * np.array([1.0, 0.9, 0.7])  # Warm tone adjustment
                arr = np.clip(arr, 0, 255)
                temp_img = Image.fromarray(arr.astype(np.uint8))
                temp_img = temp_img.filter(ImageFilter.GaussianBlur(1))  # Soft blur
                # Add subtle noise for vintage grain
                noise = np.random.normal(0, 10, arr.shape).astype(np.uint8)
                arr = np.clip(arr + noise, 0, 255)
                temp_img = Image.fromarray(arr.astype(np.uint8))
            elif f == "Oil Painting":
                cv_img2 = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2BGR)
                img_blur = cv2.xphoto.oilPainting(cv_img2, size=7, dynRatio=1)  # Enhanced oil painting effect
                temp_img = Image.fromarray(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))
            elif f == "Emboss Strong":
                temp_img = temp_img.filter(ImageFilter.EMBOSS)
                temp_img = ImageEnhance.Contrast(temp_img).enhance(2.0)  # Stronger contrast
            elif f == "Cartoon Colorful":
                cv_img2 = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.Canny(gray, 80, 150)  # Softer edges for colorful look
                edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
                edges = cv2.bitwise_not(edges)
                color = cv2.bilateralFilter(cv_img2, 7, 250, 250)
                cartoon = cv2.bitwise_and(color, color, mask=edges)
                cartoon = cv2.convertScaleAbs(cartoon, alpha=1.3, beta=30)  # Vibrant colors
                temp_img = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
            elif f == "HDR Enhanced":
                cv_img2 = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2BGR)
                hdr = cv2.detailEnhance(cv_img2, sigma_s=25, sigma_r=0.3)  # Very strong HDR
                temp_img = Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
                temp_img = ImageEnhance.Contrast(temp_img).enhance(1.5)
                temp_img = ImageEnhance.Brightness(temp_img).enhance(1.2)
            elif f == "Pencil Sketch Color":
                cv_img2 = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                inv = cv2.bitwise_not(gray)
                blur = cv2.GaussianBlur(inv, (31, 31), 0)
                inv_blur = cv2.bitwise_not(blur)
                sketch = cv2.divide(gray, inv_blur, scale=256.0)
                sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
                # Blend with original for subtle color
                blend = cv2.addWeighted(cv_img2, 0.3, sketch_rgb, 0.7, 0)
                temp_img = Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
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
[23/08, 2:40 am] Nada Salem: import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2
import io
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

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
sharpness = st.sidebar.slider("Sharpness 🔪", 0.0, 2.0, 1.0, 0.01)

# ---- Sidebar: Filters & Effects ----
st.sidebar.header("🎨 Filters & Effects")
filter_options = ["Grayscale", "Sepia", "Invert", "Blur", "Edge",
                  "Cartoon", "Emboss", "Sharpen", "Sketch", "HDR",
                  "Vintage", "Oil Painting", "Emboss Strong", "Cartoon Colorful", "HDR Enhanced", "Pencil Sketch Color"]
apply_filters = st.sidebar.multiselect("Filters 🎭", filter_options)

# ---- Sidebar: Editing Tools ----
st.sidebar.header("🛠 Editing Tools")
denoise = st.sidebar.checkbox("Denoise 🧹")
rotate_90 = st.sidebar.checkbox("Rotate 90° 🔄")
apply_crop = st.sidebar.checkbox("✂ Crop")
apply_remove = st.sidebar.checkbox("🖌 Remove")
apply_text = st.sidebar.checkbox("📝 Add Text")

# ---- File uploader ----
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

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
                denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)  # Stronger denoising
                img = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
                st.session_state.base_image img.copy()
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
            cv_img2 = cv2.cvtColor(np.array(temp_img), cv2.COLOR_RGB2BGR)
            if f == "Grayscale":
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                temp_img = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
            elif f == "Sepia":
                sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                        [0.349, 0.686, 0.168],
                                        [0.272, 0.534, 0.131]])
                sepia_img = cv2.transform(cv_img2, sepia_matrix)
                sepia_img = np.clip(sepia_img, 0, 255)
                temp_img = Image.fromarray(cv2.cvtColor(sepia_img, cv2.COLOR_BGR2RGB))
            elif f == "Invert":
                inverted = cv2.bitwise_not(cv_img2)
                temp_img = Image.fromarray(cv2.cvtColor(inverted, cv2.COLOR_BGR2RGB))
            elif f == "Blur":
                blurred = cv2.GaussianBlur(cv_img2, (15, 15), 0)  # Stronger blur
                temp_img = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
            elif f == "Edge":
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                temp_img = Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
            elif f == "Cartoon":
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.Canny(gray, 80, 150)  # Optimized for cartoon
                edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)  # Thicker outlines
                edges = cv2.bitwise_not(edges)  # Black outlines
                color = cv2.bilateralFilter(cv_img2, 9, 300, 300)  # Smooth colors
                cartoon = cv2.bitwise_and(color, color, mask=edges)
                temp_img = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
            elif f == "Emboss":
                kernel = np.array([[ -2, -1, 0],
                                  [ -1,  1, 1],
                                  [  0,  1, 2]])
                embossed = cv2.filter2D(cv_img2, -1, kernel)
                temp_img = Image.fromarray(cv2.cvtColor(embossed, cv2.COLOR_BGR2RGB))
            elif f == "Sharpen":
                kernel = np.array([[ 0, -1,  0],
                                  [ -1,  5, -1],
                                  [  0, -1,  0]])
                sharpened = cv2.filter2D(cv_img2, -1, kernel)
                temp_img = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
            elif f == "Sketch":
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.Canny(gray, 60, 120)  # Softer edges for sketch
                edges = cv2.bitwise_not(edges)
                blurred = cv2.GaussianBlur(gray, (21, 21), 0)
                sketch = cv2.divide(gray, blurred, scale=256.0)
                temp_img = Image.fromarray(cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB))
            elif f == "HDR":
                hdr = cv2.detailEnhance(cv_img2, sigma_s=30, sigma_r=0.3)  # Strong HDR
                hdr = cv2.convertScaleAbs(hdr, alpha=1.2, beta=20)  # Boost contrast/brightness
                temp_img = Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
            elif f == "Vintage":
                sepia_matrix = np.array([[0.7, 0.9, 0.5],
                                        [0.6, 0.8, 0.4],
                                        [0.5, 0.7, 0.3]])
                vintage = cv2.transform(cv_img2, sepia_matrix)
                vintage = np.clip(vintage, 0, 255)
                noise = np.random.normal(0, 15, vintage.shape).astype(np.uint8)  # Add grain
                vintage = np.clip(vintage + noise, 0, 255)
                temp_img = Image.fromarray(cv2.cvtColor(vintage, cv2.COLOR_BGR2RGB))
            elif f == "Oil Painting":
                oil = cv2.xphoto.oilPainting(cv_img2, size=7, dynRatio=1)
                temp_img = Image.fromarray(cv2.cvtColor(oil, cv2.COLOR_BGR2RGB))
            elif f == "Emboss Strong":
                kernel = np.array([[-2, -1, 0],
                                  [-1,  1, 1],
                                  [ 0,  1, 2]])
                embossed = cv2.filter2D(cv_img2, -1, kernel)
                embossed = cv2.convertScaleAbs(embossed, alpha=1.5, beta=30)  # Strong contrast
                temp_img = Image.fromarray(cv2.cvtColor(embossed, cv2.COLOR_BGR2RGB))
            elif f == "Cartoon Colorful":
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.Canny(gray, 50, 100)  # Softer edges
                edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
                edges = cv2.bitwise_not(edges)
                color = cv2.bilateralFilter(cv_img2, 7, 250, 250)
                cartoon = cv2.bitwise_and(color, color, mask=edges)
                cartoon = cv2.convertScaleAbs(cartoon, alpha=1.5, beta=50)  # Vibrant colors
                temp_img = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))
            elif f == "HDR Enhanced":
                hdr = cv2.detailEnhance(cv_img2, sigma_s=40, sigma_r=0.4)
                hdr = cv2.convertScaleAbs(hdr, alpha=1.3, beta=30)
                temp_img = Image.fromarray(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
            elif f == "Pencil Sketch Color":
                gray = cv2.cvtColor(cv_img2, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                edges = cv2.Canny(gray, 60, 120)
                edges = cv2.bitwise_not(edges)
                blurred = cv2.GaussianBlur(gray, (21, 21), 0)
                sketch = cv2.divide(gray, blurred, scale=256.0)
                sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
                blend = cv2.addWeighted(cv_img2, 0.4, sketch_rgb, 0.6, 0)  # Balanced color blend
                temp_img = Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
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