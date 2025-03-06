import streamlit as st
import cv2
import numpy as np
import json
import time
from vision_llm import VisionLLM  # Importing your class
from PIL import Image
import io

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="OCR Image Translator")

# Language Code to Language Name Mapping
LANGUAGE_MAPPING = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "ar": "Arabic",
    "ru": "Russian",
    "it": "Italian",
    "pt": "Portuguese",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "hu": "Hungarian"
}

# Initialize session state variables
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "detected_languages" not in st.session_state:
    st.session_state.detected_languages = set()
if "translated_data" not in st.session_state:
    st.session_state.translated_data = []
if "paragraphs" not in st.session_state:
    st.session_state.paragraphs = []
if "lang_detection_time" not in st.session_state:
    st.session_state.lang_detection_time = None  
if "translation_time" not in st.session_state:
    st.session_state.translation_time = None  
if "selected_translation" not in st.session_state:
    st.session_state.selected_translation = None  

# Title
st.title("📄 OCR Image Translator")

# Sample Images - Add your images to a specific directory
sample_images = [
    "aeraa.jpg",
    "hin.jpg",
    "hung.png",
    "jap.png",
    'par.jpeg',
    'fren.jpg',
    'ara1.png',
    'araee1.png'
]

# Image Selection Dropdown
selected_image = st.selectbox("Select an Image", ["Select an image"] + sample_images)

# Load the selected image
if selected_image != "Select an image":
    # Reset translation-related states when a new image is uploaded
    st.session_state.selected_translation = None
    st.session_state.processed_image = None
    st.session_state.translated_data = []
    st.session_state.lang_detection_time = None
    st.session_state.translation_time = None

    image_path = f"sample_images/{selected_image}"  # Assuming images are in an 'images' folder
    st.session_state.image_path = image_path  # Save the image path for later use

    # Load the image using PIL (streamlit uploads use PIL images)
    image = Image.open(image_path)
    
    # Convert the PIL image to a numpy array (OpenCV format)
    image_np = np.array(image)
    
    # Convert RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Save the image as a temporary file
    image_path = "uploaded_image.jpg"
    cv2.imwrite(image_path, image_bgr)
    
    # Initialize OCR object
    ocr_translator = VisionLLM(image_path)

    # Preprocess Image
    orig_image, _ = ocr_translator.preprocess_image(image_path)

    # Extract paragraphs
    paragraphs = ocr_translator.extract_paragraphs_with_bounding_boxes(orig_image)

    # **Start timer for language detection**
    start_time = time.time()

    # Detect language
    detected_languages = set()
    for text, (x, y, w, h), _ in paragraphs:
        cropped_image = orig_image[y:y+h, x:x+w]  
        detected_lang_code = ocr_translator.qwen(cropped_image)  
        
        # Map language code to language name
        detected_lang_name = LANGUAGE_MAPPING.get(detected_lang_code, detected_lang_code)  # Default to code if not found
        detected_languages.add(detected_lang_name)

    # **End timer for language detection**
    st.session_state.lang_detection_time = round(time.time() - start_time, 2)

    # Update session state
    st.session_state.image_uploaded = True
    st.session_state.detected_languages = detected_languages
    st.session_state.paragraphs = paragraphs

# Show detected language **only after an image is uploaded**
if st.session_state.image_uploaded:
    st.info(f"**Detected Language(s):** {', '.join(st.session_state.detected_languages) if st.session_state.detected_languages else 'N/A'}")
    
    if st.session_state.lang_detection_time is not None:
        st.write(f"⏳ **Language Detection Time:** {st.session_state.lang_detection_time} seconds")

    # **Show translation buttons after image selection**
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Translate with Qwen"):
            st.session_state.selected_translation = "qwen"
            st.session_state.processed_image = None  # Clear previous output
    with col2:
        if st.button("🌍 Translate with DeepL"):
            st.session_state.selected_translation = "deepl"
            st.session_state.processed_image = None  # Clear previous output

# Display uploaded image (Left Side)
if st.session_state.image_uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(st.session_state.image_path, caption="Original Image", use_container_width=True)

    # Process Translation ONLY if a translation method is selected
    if st.session_state.selected_translation:
        translated_data = []
        translated_texts = []

        # **Start timer for translation**
        start_time = time.time()

        for text, (x, y, w, h), _ in st.session_state.paragraphs:
            cropped_image = orig_image[y:y+h, x:x+w]  
            detected_lang_code = ocr_translator.qwen(cropped_image)  
            
            # Map language code to language name
            detected_lang_name = LANGUAGE_MAPPING.get(detected_lang_code, detected_lang_code)

            if detected_lang_code != "en":
                if st.session_state.selected_translation == "qwen":
                    translated_text = ocr_translator.qwen_translate_to_english(cropped_image, text)
                elif st.session_state.selected_translation == "deepl":
                    translated_text = ocr_translator.deepl_translate_to_english(text, detected_lang_code)
            else:
                translated_text = text

            translated_texts.append(translated_text)

            # Append to JSON output
            translated_data.append({
                "original_text": text,
                "translated_text": translated_text
            })

        # **End timer for translation**
        st.session_state.translation_time = round(time.time() - start_time, 2)

        # Replace text with translation in image
        output_image_path, _ = ocr_translator.replace_text_with_translation(
            st.session_state.image_path, st.session_state.paragraphs, translated_texts
        )

        # Load processed image
        processed_image = cv2.imread(output_image_path)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Store results in session state
        st.session_state.processed_image = processed_image
        st.session_state.translated_data = translated_data

# Show results only after translation
if st.session_state.processed_image is not None:
    # Display Processed Image (Right Side)
    with col2:
        st.subheader("📷 Translated Image")
        st.image(st.session_state.processed_image, caption="Translated Image", use_container_width=True)

    if st.session_state.translation_time is not None:
        st.write(f"⏳ **Translation Time:** {st.session_state.translation_time} seconds")
