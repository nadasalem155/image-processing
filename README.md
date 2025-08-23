# 🎨 image-processing  

"Interactive Image Editor – Python project available as a Jupyter Notebook and a standalone .py file. Apply various filters, rotate, crop, remove objects, and enhance images interactively within the notebook or as a web application."  

🔗 [Try the Web App here](https://image-processing5.streamlit.app/)  

---

## ✨ Features  

### 📝 Notebook (image-editing/editing.ipynb)  
The Jupyter Notebook contains step-by-step implementations of different image processing techniques using OpenCV and PIL. Each function demonstrates its effect clearly:  

1. 📏 **Resize** – change image dimensions while keeping proportions or setting custom width/height.  
2. 🎨 **Filters** – apply predefined artistic filters.  
3. 🧹 **Denoising** – remove noise and smooth images using OpenCV’s `fastNlMeansDenoisingColored`.  
4. 🔆 **Brightness & Contrast** – adjust image lightness and contrast values numerically.  
5. 🔄 **Rotate, Flip & Crop** – enter custom values (angles, flip axis, crop dimensions) for precise transformations.  
6. 🌈 **Adjust Hue** – modify color tones by shifting hue values.  
7. 💎 **Gamma Correction** – non-linear brightness adjustment to lighten or darken naturally.  
8. ⚡ **CLAHE Color Equalization** – enhance local contrast and details using Contrast Limited Adaptive Histogram Equalization.  
9. 🖌️ **Oil Painting Effect** – simulate brush-stroke style painting.  
10. 💧 **Watercolor Effect** – create a smooth watercolor-like look.  
11. 🕰️ **Vintage Effect** – add retro style tones to the image.  
12. 😊 **Face Beautification** – smooth skin and enhance facial features for portraits.  
13. 🧽 **Remove Objects** – erase unwanted areas by selecting regions and replacing them with surrounding pixels.  

All results from the notebook are saved into the `RESULT/` folder.  

---

### 🌐 Web Application (app.py)  
The Streamlit web app provides an **interactive UI** for image editing with the following features:  
- 📤 **Upload an Image** – import any photo.  
- 📏 **Resize & Crop** – change dimensions or crop areas directly in the browser.  
- 🔄 **Rotate & Flip** – adjust orientation instantly.  
- 🎨 **Filters** – cartoon, vintage, oil painting, watercolor, and more.  
- 🧹 **Denoising** – remove unwanted noise.  
- 🔆 **Brightness & Contrast** – sliders to enhance or reduce tone.  
- 🌈 **Adjust Hue & Saturation** – control colors dynamically.  
- 💎 **Gamma Correction** – improve lighting.  
- ⚡ **CLAHE Equalization** – enhance contrast in specific regions.  
- 😊 **Face Beautification** – one-click beautify option.  
- 🧽 **Remove Objects** – paint over objects to erase them.  
- ✍️ **Add Text & Emojis** – personalize images with captions or fun icons (web-only feature).  

---

## 📂 Project Structure

image-processing/ │-- app.py                  # Streamlit web application │-- requirements.txt        # Required dependencies │-- README.md               # Project documentation │ │-- image-editing/          # Jupyter Notebook folder │   └── editing.ipynb       # Notebook for step-by-step editing │ │-- images/                 # Input images used in the notebook │-- RESULT/                 # Output results generated from the notebook

---

## 🚀 How to Run  

### ▶️ Notebook  
1. Open `image-editing/editing.ipynb` in Jupyter Notebook.  
2. Run the cells sequentially to apply transformations.  
3. Input images are in `images/` and edited results will be saved in `RESULT/`.  

### 🌐 Web App  
1. Install the dependencies:  
   ```bash
   pip install -r requirements.txt

2. Run the app:

streamlit run app.py


3. Open the provided local URL in your browser.


4. Upload an image and start editing interactively.




---

🎯 Use Cases

Quick photo enhancements and artistic effects.

Learning and experimenting with OpenCV & PIL functions.

Foundation for more advanced AI-powered editing projects.



---

📌 Notes

Input samples are available in the images/ folder.

Notebook results are stored in RESULT/.

Web app provides a beginner-friendly interface for editing without coding.
