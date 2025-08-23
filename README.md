# 🎨 image-processing  

"Interactive Image Editor – Python project available as a Jupyter Notebook and a standalone .py file. Apply various filters, rotate, crop, remove objects, and enhance images interactively within the notebook or as a web application."  

🔗 [Try the Web App here](https://image-processing5.streamlit.app/)  

---

## ✨ Features  

This project comes in **two versions**:  

---

### 📝 Notebook (image-editing/editing.ipynb)  
The Jupyter Notebook contains **step-by-step implementations** of image editing techniques using OpenCV and PIL.  
Available functions:  

1. 📏 **Resize** – change image dimensions with custom width/height.  
2. 🎨 **Filters** – apply different artistic filters.  
3. 🧹 **Denoising** – smooth and remove unwanted noise.  
4. 🔆 **Brightness & Contrast** – adjust image tone numerically.  
5. 🔄 **Rotate, Flip & Crop** – transform image orientation or crop by coordinates.  
6. 🌈 **Adjust Hue** – shift color tones across the spectrum.  
7. 💎 **Gamma Correction** – nonlinear brightness enhancement.  
8. ⚡ **CLAHE Color Equalization** – boost local contrast with adaptive histogram equalization.  
9. 🖌️ **Oil Painting Effect** – simulate a painted look with brush strokes.  
10. 💧 **Watercolor Effect** – smooth watercolor-like effect.  
11. 🕰️ **Vintage Effect** – apply retro tones.  
12. 😊 **Face Beautification** – smooth skin and enhance portrait quality.  
13. 🧽 **Remove Objects** – erase selected regions by filling from nearby pixels.  

👉 Results are automatically saved in the `RESULT/` folder.  

---

### 🌐 Web Application (app.py)  
The Streamlit app provides a **user-friendly interface** with real-time editing.  

**Editing Tools**  
- 📤 Upload images (`.jpg`, `.jpeg`, `.png`).  
- ✂ **Crop** images interactively (drag box).  
- 🔄 **Rotate 90°** with one click.  
- 🧹 **Denoise** noisy images.  
- 📝 **Add Text** with custom size, color, and positioning.  
- ↩ **Undo** history of edits.  
- 💾 **Download** final image.  

**Adjustments (via sliders)**  
- ☀ **Brightness**  
- 🎚 **Contrast**  
- 🔪 **Sharpness**  

**Filters & Effects**  
- ⚫ **Grayscale**  
- 🤎 **Sepia**  
- 💨 **Blur**  
- 🎭 **Cartoon**  
- 🌈 **Cartoon Colorful**  
- ✨ **HDR Enhanced**  

---

## 📂 Project Structure

image-processing/ │-- app.py                  # Streamlit web application │-- requirements.txt        # Required dependencies │-- README.md               # Project documentation │ │-- image-editing/          # Jupyter Notebook folder │   └── editing.ipynb       # Notebook for step-by-step editing │ │-- images/                 # Input images used in the notebook │-- RESULT/                 # Output results generated from the notebook

---

## 🚀 How to Run  

### ▶️ Notebook  
1. Open `image-editing/editing.ipynb` in Jupyter Notebook.  
2. Run cells sequentially to apply transformations.  
3. Input images go in `images/`, results appear in `RESULT/`.  

### 🌐 Web App  
1. Install the dependencies:  
   ```bash
   pip install -r requirements.txt

2. Run the app:

streamlit run app.py


3. Open the local URL in your browser to start editing interactively.




---

🎯 Use Cases

Quick artistic photo edits.

Learning and experimenting with OpenCV & PIL functions.

Interactive image editor with both coding (Notebook) and no-code (Web App) options.



---
