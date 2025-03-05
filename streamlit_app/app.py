import streamlit as st
import numpy as np
from PIL import Image
from utils.image_preprocessing import preprocess_image
from utils.text_extraction import extract_paragraphs_with_bounding_boxes
from utils.language_detection import detect_language
from utils.text_translation import translate_text
from utils.text_replacement import replace_text_with_translation
import pytesseract

# **Modern Page Configuration**
st.set_page_config(page_title="OCR Image Translator", layout="wide")


# **Custom CSS for JS-like Modern UI**
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .header-container {
        text-align: center;
        padding: 20px 0;
        font-size: 2.5rem;
        font-weight: bold;
        color: #0d6efd;
    }
    .upload-section {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        padding: 20px;
    }
    .upload-box {
        border: 2px dashed #0d6efd;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        width: 400px;
        background-color: #f8f9fa;
    }
    .image-container {
        text-align: center;
        margin-top: 10px;
    }
    .image-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: #0d6efd;
    }
    .process-btn {
        background-color: #0d6efd;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-size: 1rem;
    }
    .process-btn:hover {
        background-color: #084298;
    }
    </style>
""", unsafe_allow_html=True)

# **Header**
st.markdown('<p class="header-container">📄 OCR Image Translator</p>', unsafe_allow_html=True)

# **Upload Section**
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# **Process Button (Always on Top)**
if uploaded_file:
    st.markdown('<div style="text-align: center; margin-top: 10px;">', unsafe_allow_html=True)
    process_button = st.button("🔍 Process Image", key="process_btn")
    st.markdown('</div>', unsafe_allow_html=True)

# **Images Display (Side by Side)**
col_left, col_right = st.columns(2)

if uploaded_file:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Show Uploaded Image (Left)
    with col_left:
        st.markdown('<p class="image-header">📷 Uploaded Image</p>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)

    # Process and Show Result (Right)
    if process_button:
        preprocessed_image = preprocess_image(image)
        paragraphs_with_boxes = extract_paragraphs_with_bounding_boxes(preprocessed_image)
        detected_languages = {p: detect_language(p) for p, _ in paragraphs_with_boxes}
        translated_texts = [translate_text(p) if detected_languages[p] != "en" else p for p, _ in paragraphs_with_boxes]

        result = replace_text_with_translation(preprocessed_image, paragraphs_with_boxes, translated_texts)
        processed_image = result[0] if isinstance(result, tuple) else result

        with col_right:
            st.markdown('<p class="image-header">✅ Processed Image</p>', unsafe_allow_html=True)
            st.image(processed_image, channels="BGR", use_container_width=True)
