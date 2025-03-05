import streamlit as st
import cv2
import numpy as np
from utils.image_preprocessing import preprocess_image
from utils.text_extraction import extract_paragraphs_with_bounding_boxes
from utils.language_detection import detect_language
from utils.text_translation import translate_text  # DeepL translation
from utils.qwen_translate import qwen_translate_to_english  # Qwen translation
from utils.text_replacement import replace_text_with_translation

# --- Title & Upload Box (Fixed at Top) ---
st.title("Multilingual OCR Translator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="file_uploader")

# --- Button Section (Fixed Below Upload Box) ---
col1, col2 = st.columns(2)  # Keep buttons evenly spaced
with col1:
    process_button = st.button("Translate using DeepL")  # Uses translate_text()
with col2:
    process_qwen_button = st.button("Translate using Qwen")  # Uses qwen_translate_to_english()

# --- Process Image ---
if uploaded_file:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # **Ensure FULL-SIZED images without overlap**
    col_left, col_right = st.columns(2)  # Evenly divide the screen

    with col_left:
        st.image(image, caption="Uploaded Image", use_column_width=True)  # Ensure full width

    if process_button or process_qwen_button:
        preprocessed_image = preprocess_image(image)
        paragraphs_with_boxes = extract_paragraphs_with_bounding_boxes(preprocessed_image)

        detected_languages = {p: detect_language(p) for p, _ in paragraphs_with_boxes}

        translated_texts = []
        for p, bbox in paragraphs_with_boxes:
            if detected_languages[p] != "en":
                if process_button:
                    translated_texts.append(translate_text(p))  # DeepL translation
                elif process_qwen_button:
                    translated_text, success = qwen_translate_to_english(p)
                    translated_texts.append(translated_text if success else p)  # Fallback if API fails
            else:
                translated_texts.append(p)  # Keep English text unchanged

        result = replace_text_with_translation(preprocessed_image, paragraphs_with_boxes, translated_texts)
        processed_image = result[0] if isinstance(result, tuple) else result

        with col_right:
            st.image(processed_image, caption="Translated Image", use_column_width=True)  # Ensure full width