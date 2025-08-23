# 🎨 image-processing  

"Interactive Image Editor – Python project available as a Jupyter Notebook and a standalone .py file. Apply various filters, rotate, crop, remove objects, add text and emojis, and enhance images interactively within the notebook or as a web application."  

🔗 [Try the Web App here](https://image-processing5.streamlit.app/)  

---

## ✨ Features  

### 📝 Notebook (image-editing/editing.ipynb)  
The Jupyter Notebook demonstrates how each image-processing technique works step by step. You can:  
- 🖼️ **Crop & Resize** – manually select regions of interest or scale images.  
- 🔄 **Rotate & Flip** – rotate images by angles or flip horizontally/vertically.  
- 🧹 **Denoising** – reduce image noise using OpenCV filters.  
- 🌈 **Color Adjustments** – change hue, saturation, and lightness levels.  
- ⚡ **Histogram Equalization** – improve contrast (with and without CLAHE).  
- 🎭 **Apply Filters** – cartoon effect, vintage, oil painting, watercolor, etc.  
- ✍️ **Add Text & Emojis** – overlay text or fun emojis on images.  
- 💎 **Gamma Correction** – adjust brightness in a non-linear way.  
- 😊 **Face Beautification** – smoothen skin and enhance facial features.  
- 📂 **Save Results** – edited outputs are stored in the `RESULT/` folder.  

Great for learning how each transformation works programmatically with OpenCV and PIL.  

---

### 🌐 Web Application (app.py)  
The web version is built with **Streamlit**, making it easy to edit images interactively:  
- 📤 **Upload an Image** – load your own photo to start editing.  
- 🎨 **Adjust Hue & Saturation** – fine-tune colors to your preference.  
- 🔆 **Brightness & Contrast Control** – enhance visibility and tone.  
- 🧹 **Denoising** – smooth noisy images.  
- ✂️ **Resize & Crop** – cut or scale images directly in the browser.  
- 🔄 **Rotate & Flip** – orientation adjustments with a single click.  
- 🎭 **Creative Filters** – cartoon, oil painting, watercolor, vintage.  
- 💎 **Gamma Correction** – adjust lighting naturally.  
- 😊 **Face Beautification** – quick beauty enhancements.  
- 🧽 **Object Removal** – erase unwanted regions by drawing over them.  
- 😃 **Add Text & Emojis** – personalize with captions and icons.  

The web app is user-friendly and requires no coding knowledge.  

---

## 📂 Project Structure

image-processing/ │-- app.py                  # Streamlit web application │-- requirements.txt        # Required dependencies │-- README.md               # Project documentation │ │-- image-editing/          # Jupyter Notebook folder │   └── editing.ipynb       # Notebook for step-by-step editing │ │-- images/                 # Input images used in the notebook │-- RESULT/                 # Output results generated from the notebook

---

## 🚀 How to Run  

### ▶️ Notebook  
1. Open `image-editing/editing.ipynb` in Jupyter Notebook.  
2. Run the cells to apply filters or transformations.  
3. Input images are in `images/` and edited results will be saved to `RESULT/`.  

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

Quick photo touch-ups.

Learning and experimenting with OpenCV & PIL.

Building a foundation for advanced AI-based image editing projects.



---

📌 Notes

Input samples are available in the images/ folder.

All edited outputs from the notebook will be stored in RESULT/.

The web app does not require coding skills—just upload and edit.