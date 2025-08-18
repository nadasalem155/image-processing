# 🖼️ Interactive Image Processing & Editing  

An **interactive image editor** built with **Python**.  
Available as both a **Jupyter Notebook** and a **Streamlit Web Application**.  

Easily apply filters, rotate, crop, remove objects, adjust image properties, and more — all interactively.  

---

## 🚀 Features  

### 🔹 Jupyter Notebook Version  
The notebook provides step-by-step image editing functionalities where users can:  
- Adjust **hue, saturation, brightness, contrast**.  
- Apply **CLAHE color enhancement**.  
- **Resize** images.  
- **Denoise** images.  
- Apply different **filters** by entering their number (e.g., blur, sharpen, edge detection).  
- Perform **rotate, crop, and flip (vertical/horizontal)** operations.  
- Apply effects:  
  - Gamma correction  
  - Oil painting effect  
  - Watercolor effect  
  - Vintage effect  
  - Face beautification  
- **Object removal** by manually selecting unwanted parts.  

---

### 🌐 Streamlit Web Application  
Try the interactive web app here:  
👉 [Live App Link](PUT-YOUR-LINK-HERE)  

#### How to Use:  
1. **Upload Image** – choose any image from your device.  
2. **Crop** – select the crop option and drag over the area to keep.  
3. **Rotate** – enter a rotation angle (e.g., 90, 180).  
4. **Flip** – choose **Vertical** or **Horizontal** flip.  
5. **Adjustments** – brightness, contrast, hue, saturation, CLAHE.  
6. **Filters & Effects** – oil painting, watercolor, vintage, gamma correction, beautification.  
7. **Object Removal** – select the eraser tool and draw over unwanted objects.  
8. **Text & Emojis** – add custom text or emojis.  
9. **Download Result** – once satisfied, download the final edited image.  

All edits are previewed live in the browser.  

---

## ⚙️ Installation (Local)  

Clone the repository and install requirements:  

```bash
git clone https://github.com/yourusername/image-processing.git
cd image-processing
pip install -r requirements.txt