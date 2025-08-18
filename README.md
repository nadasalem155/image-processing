# 🎨 image-processing  
"Interactive Image Editor – Python project available as a Jupyter Notebook and a standalone .py file. Apply various filters, rotate, crop, remove objects, add text and emojis, and enhance images interactively within the notebook or as a web application."

---

🌍 **Live Web App:** [Open Image Editor](https://image-processing5.streamlit.app/)

---

## 📂 Project Structure

image-processing/ ├── app.py                # Streamlit web application 
                  ├── requirements.txt      # Required dependencies
                  ├── image-editing/        # Folder containing the Jupyter notebook 
                     │└── editing.ipynb     # Notebook with image editing operations
                     ├── images/               # Sample images used inside the notebook 
                     └── RESULT/               # Output images generated from the notebook

---

## ⚡ Features

This project provides both **a Jupyter Notebook** and **a Web Application (Streamlit)** for interactive image editing.  
You can apply the following operations:

- 🎨 **Adjustments**: Hue, Saturation, Contrast, Brightness, Gamma correction, CLAHE color enhancement  
- 🖼 **Transformations**: Rotate (choose angle), Flip (horizontal/vertical), Crop, Resize  
- ✨ **Effects**: Oil painting effect, Watercolor effect, Vintage filter  
- 👩 Face beautification  
- 🧽 Denoising  
- 🎭 Filters: Apply by selecting a filter number to transform the image  
- 🖌 Object removal (draw over the part to remove)  
- 🔤 Add text and emojis  
- 📐 Combine multiple operations (e.g., rotate + crop together)  

---

## 🖥️ How to Use the Web App

When you open the [Live Web App](https://your-app-link-here.com):

1. **Upload an image** from your device.  
2. Choose the operation you want to apply:  
   - **Rotate** → enter the degree of rotation.  
   - **Flip** → choose horizontal or vertical.  
   - **Crop** → select the area to keep.  
   - **Filters / Effects** → pick the filter or effect number.  
   - **Brightness / Contrast / Hue** → adjust sliders to enhance image.  
   - **Remove Objects** → draw on the part you want to remove.  
3. Preview the edited image directly.  
4. Save the result if you like it.  

---

## 📓 How to Use the Notebook

- Open the Jupyter Notebook inside `image-editing/editing.ipynb`.  
- Each section contains ready-to-run cells for:  
  - Adjustments (contrast, brightness, hue, gamma correction)  
  - Applying filters and effects  
  - Transformations (rotate, crop, flip, resize)  
  - Advanced enhancements (denoising, face beautification, object removal)  
- Run the cells step by step to see outputs stored in the **RESULT/** folder.  

---
