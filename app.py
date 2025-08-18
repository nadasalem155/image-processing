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

# ----------------- Session state -----------------
if "orig_image" not in st.session_state:
    st.session_state.orig_image = None
if "working_image" not in st.session_state:
    st.session_state.working_image = None
if "history" not in st.session_state:
    st.session_state.history = []
if "apply_crop" not in st.session_state:
    st.session_state.apply_crop = False

# ----------------- Upload image -----------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.session_state.orig_image = img
    st.session_state.working_image = img.copy()
    st.session_state.history = [img.copy()]
    st.session_state.apply_crop = False

# ----------------- Helper functions -----------------
def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def try_load_font(size=24):
    try:
        return ImageFont.truetype("seguiemj.ttf", size)
    except:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except:
            try:
                return ImageFont.truetype("DejaVuSans.ttf", size)
            except:
                st.warning("No suitable font found. Emojis may not render correctly.")
                return ImageFont.load_default()

def get_mobile_dimensions(pil_img, max_width=350):
    aspect_ratio = pil_img.height / pil_img.width
    width = min(pil_img.width, max_width)
    height = int(width * aspect_ratio)
    return width, height

# ----------------- Sidebar Tools -----------------
st.sidebar.header("🛠 Tools")

filter_choice = st.sidebar.selectbox("🎨 Filter", [
    "None", "Grayscale", "Sepia", "Invert", "Blur", "Edge", 
    "Cartoon", "Emboss", "Sharpen", "Sketch", "HDR",
    "Vintage", "Oil Painting", "Emboss Strong", "Cartoon Colorful", "HDR Enhanced", "Pencil Sketch Color"
])

brightness = st.sidebar.slider("☀ Brightness", -100, 100, 0)
contrast = st.sidebar.slider("🌗 Contrast", 0.5, 3.0, 1.0)
sharpness = st.sidebar.slider("🔪 Sharpness", 0.5, 3.0, 1.0)
denoise = st.sidebar.slider("🧹 Noise Removal", 0, 30, 0)

rotate_choice = st.sidebar.radio("🔄 Rotate", ["None", "90°", "180°", "270°", "360° Reset"])

do_crop = st.sidebar.checkbox("✂ Crop Mode", key="crop_mode", value=st.session_state.apply_crop)
do_remove = st.sidebar.checkbox("🧽 Remove Mode")
add_texts = st.sidebar.text_area("📝 Add Texts (one per line)", "").split("\n")
add_texts = [text.strip() for text in add_texts if text.strip()]
text_size = st.sidebar.slider("📏 Text Size", 12, 100, 36)
text_color = st.sidebar.color_picker("🎨 Text Color", "#FFFF00")

add_emojis = st.sidebar.multiselect("😂 Emojis", ["😂", "😍", "🔥", "👍", "❤", "😎"], default=[])
emoji_size = st.sidebar.slider("📏 Emoji Size", 12, 100, 36)

# ----------------- Apply Filters and Enhancements -----------------
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
            kernel = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
            cv_img = cv2.transform(cv_img, kernel)
            cv_img = np.clip(cv_img,0,255).astype(np.uint8)
        elif filter_choice == "Invert":
            cv_img = cv2.bitwise_not(cv_img)
        elif filter_choice == "Blur":
            cv_img = cv2.GaussianBlur(cv_img,(15,15),0)
        elif filter_choice == "Edge":
            edges = cv2.Canny(cv_img,100,200)
            cv_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif filter_choice == "Cartoon":
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray,5)
            edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,9)
            color = cv2.bilateralFilter(cv_img,9,250,250)
            cv_img = cv2.bitwise_and(color,color,mask=edges)
        elif filter_choice == "Emboss":
            kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
            cv_img = cv2.filter2D(cv_img,-1,kernel)+128
        elif filter_choice == "Sharpen":
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
            cv_img = cv2.filter2D(cv_img,-1,kernel)
        elif filter_choice == "Sketch":
            gray = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv,(21,21),0)
            sketch = cv2.divide(gray,255-blur,scale=256)
            cv_img = cv2.cvtColor(sketch,cv2.COLOR_GRAY2BGR)
        elif filter_choice == "HDR":
            cv_img = cv2.detailEnhance(cv_img,sigma_s=12,sigma_r=0.15)
        elif filter_choice=="Vintage":
            hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.add(hsv[:,:,1], -30)
            cv_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif filter_choice=="Oil Painting":
            cv_img = cv2.xphoto.oilPainting(cv_img,7,1)
        elif filter_choice=="Emboss Strong":
            kernel = np.array([[-4,-2,0],[-2,1,2],[0,2,4]])
            cv_img = cv2.filter2D(cv_img,-1,kernel)+128
        elif filter_choice=="Cartoon Colorful":
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray,5)
            edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,7)
            color = cv2.bilateralFilter(cv_img,9,300,300)
            cv_img = cv2.bitwise_and(color,color,mask=edges)
        elif filter_choice=="HDR Enhanced":
            cv_img = cv2.detailEnhance(cv_img,sigma_s=16,sigma_r=0.2)
        elif filter_choice=="Pencil Sketch Color":
            gray = cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv,(15,15),0)
            sketch = cv2.divide(gray,255-blur,scale=256)
            color = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
            cv_img = cv2.multiply(cv2.cvtColor(sketch,cv2.COLOR_GRAY2BGR)/255.0, color)
            cv_img = np.clip(cv_img,0,255).astype(np.uint8)
    except Exception as e:
        st.warning(f"Filter error: {e}")

    # Brightness & Contrast
    cv_img = cv2.convertScaleAbs(cv_img, alpha=contrast, beta=brightness)

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

# ----------------- Draw Canvas -----------------
if st.session_state.working_image:
    apply_edits()
    img = st.session_state.working_image.copy()

    # تحويل الصورة إلى PNG لضمان التوافق
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_png = Image.open(buf)

    # عرض الصورة للتصحيح
    st.image(img_png, caption="Debug: Background Image for Canvas", use_column_width=False)

    st.write("✏ Draw on Image")

    # Determine drawing mode
    if do_crop or add_texts or add_emojis:
        drawing_mode = "rect"
    elif do_remove:
        drawing_mode = "freedraw"
    else:
        drawing_mode = "transform"

    # Mobile-friendly size
    canvas_width, canvas_height = get_mobile_dimensions(img)

    canvas_result = st_canvas(
        fill_color="rgba(255,0,0,0.3)",
        stroke_width=5,
        stroke_color="red",
        background_image=img_png,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        key=f"canvas_{str(hash(str(img)))}"  # مفتاح ديناميكي
    )

    # تصحيح: طباعة بيانات الـ canvas
    st.write("Debug: Canvas JSON Data", canvas_result.json_data)
    st.write("Debug: Canvas Image Data Shape", canvas_result.image_data.shape if canvas_result.image_data is not None else "No image data")

    # ----------------- Apply Changes -----------------
    if st.button("Apply Changes"):
        modified = False
        current_img = st.session_state.working_image.copy()

        # Crop
        if do_crop and canvas_result.json_data and len(canvas_result.json_data["objects"])>0:
            obj = canvas_result.json_data["objects"][0]
            if obj["type"]=="rect":
                scale_x = img.width / canvas_width
                scale_y = img.height / canvas_height
                left = max(0,int(obj["left"]*scale_x))
                top = max(0,int(obj["top"]*scale_y))
                right = min(img.width,int((obj["left"]+obj["width"])*scale_x))
                bottom = min(img.height,int((obj["top"]+obj["height"])*scale_y))
                if right>left and bottom>top:
                    current_img = current_img.crop((left,top,right,bottom))
                    modified = True
                    st.session_state.apply_crop = False
                    st.success("Image cropped successfully!")
            canvas_result.json_data["objects"] = []  # clear rectangle after apply

        # Remove objects
        if do_remove and canvas_result.image_data is not None:
            mask = canvas_result.image_data[:,:,3]>0
            mask = mask.astype(np.uint8)*255
            mask = cv2.resize(mask,(current_img.width,current_img.height),interpolation=cv2.INTER_NEAREST)
            cv_img = pil_to_cv2(current_img)
            try:
                inpainted = cv2.inpaint(cv_img,mask,3,cv2.INPAINT_TELEA)
                current_img = cv2_to_pil(inpainted)
                modified = True
                st.success("Object removed successfully!")
            except Exception as e:
                st.error(f"Inpainting error: {e}")

        # Add texts and emojis
        if (add_texts or add_emojis) and canvas_result.json_data and len(canvas_result.json_data["objects"])>0:
            draw = ImageDraw.Draw(current_img)
            font = try_load_font(size=max(text_size,emoji_size))
            text_index, emoji_index = 0,0
            for obj in canvas_result.json_data["objects"]:
                if obj["type"]=="rect":
                    scale_x = img.width / canvas_width
                    scale_y = img.height / canvas_height
                    left = int(obj["left"]*scale_x)
                    top = int(obj["top"]*scale_y)
                    current_top = top

                    if text_index<len(add_texts):
                        draw.text((left,current_top),add_texts[text_index],fill=text_color,font=font)
                        current_top += text_size
                        text_index+=1
                        modified = True
                    if emoji_index<len(add_emojis):
                        draw.text((left,current_top),add_emojis[emoji_index],fill=text_color,font=font)
                        current_top += emoji_size
                        emoji_index+=1
                        modified = True

            canvas_result.json_data["objects"] = []  # clear after apply

        if modified:
            st.session_state.working_image = current_img
            st.session_state.history.append(current_img.copy())

    # Undo
    if st.button("Undo") and len(st.session_state.history)>1:
        st.session_state.history.pop()
        st.session_state.working_image = st.session_state.history[-1].copy()

    # Display final image
    final_width, final_height = get_mobile_dimensions(st.session_state.working_image)
    st.image(st.session_state.working_image, use_column_width=False, width=final_width)

    # Save
    buf = io.BytesIO()
    st.session_state.working_image.save(buf, format="PNG")
    st.download_button("💾 Download Edited Image", data=buf.getvalue(), file_name="edited.png", mime="image/png")