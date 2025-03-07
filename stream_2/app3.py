import streamlit as st
import cv2
import numpy as np
import time
from vision_llm import VisionLLM  # Importing your class
from PIL import Image
from segmentation.paragraph_detector import extract_paragraphs

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

# Sample images for selection
sample_images = ["hin.jpg", "jap.png", 'par.jpeg', 'fren.jpg', 'araa.png']

# Google Lens Translated Images
google_lens_images = {
    "hin.jpg": "lens_hin.jpg",
    "jap.png": "lens_jap.JPG",
    "par.jpeg": "lens_par.JPG",
    "fren.jpg": "lens_fre.JPG",
    "araa.png": "lens_arab.JPG"
}

# ✅ Initialize session state variables if not already set
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False
if "input_mode" not in st.session_state:
    st.session_state.input_mode = None
if "selected_translation" not in st.session_state:
    st.session_state.selected_translation = None
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
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "show_google_lens" not in st.session_state:
    st.session_state.show_google_lens = False

# Title
st.title("📄 OCR Image Translator")

# Input Method Selection (Dropdown or Upload)
input_method = st.radio("Choose Input Method", ["Select from Sample", "Upload Image"])

# 🔄 Reset session state when switching input modes
if st.session_state.input_mode != input_method:
    st.session_state.image_uploaded = False
    st.session_state.selected_translation = None
    st.session_state.processed_image = None
    st.session_state.translated_data = []
    st.session_state.lang_detection_time = None
    st.session_state.translation_time = None
    st.session_state.show_google_lens = False
    st.session_state.image_path = None
    st.session_state.input_mode = input_method  # Update input mode

# 🎯 Image Selection (Dropdown)
if input_method == "Select from Sample":
    selected_image = st.selectbox("Select an Image", ["Select an image"] + sample_images)

    if selected_image != "Select an image":
        st.session_state.image_path = f"sample_images/{selected_image}"
        st.session_state.show_google_lens = True  # Show Google Lens Image for verification
        st.session_state.image_uploaded = True
    

# 📤 Image Upload
elif input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        # Convert uploaded image to OpenCV format
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Save uploaded image
        st.session_state.image_path = "uploaded_image.jpg"
        cv2.imwrite(st.session_state.image_path, image_bgr)

        # Uploaded images don't have Google Lens verification
        st.session_state.show_google_lens = False
        st.session_state.image_uploaded = True

# 🖼️ Process Image after Selection/Upload
if st.session_state.image_uploaded and st.session_state.image_path:
    image = Image.open(st.session_state.image_path)

    # Initialize OCR object
    ocr_translator = VisionLLM(st.session_state.image_path)

    # Preprocess Image
    orig_image, _ = ocr_translator.preprocess_image(st.session_state.image_path)

    # Extract paragraphs
    text_bboxes = ocr_translator.extract_text(image)
    paragraphs = extract_paragraphs(image, texts_b_boxes=text_bboxes)

    # 🕒 Start Language Detection Timer
    start_time = time.time()

    # Detect language
    detected_languages = set()
    for text, (x, y, w, h), _ in paragraphs:
        cropped_image = orig_image[y:y+h, x:x+w]
        detected_lang_code = ocr_translator.qwen(cropped_image)
        detected_lang_name = LANGUAGE_MAPPING.get(detected_lang_code, detected_lang_code)
        detected_languages.add(detected_lang_name)

    # 🕒 End Language Detection Timer
    st.session_state.lang_detection_time = round(time.time() - start_time, 2)

    # Update session state
    st.session_state.detected_languages = detected_languages
    st.session_state.paragraphs = paragraphs

# 🔍 Show Detected Language
if st.session_state.image_uploaded:
    st.info(f"**Detected Language(s):** {', '.join(st.session_state.detected_languages) if st.session_state.detected_languages else 'N/A'}")

    if st.session_state.lang_detection_time:
        st.write(f"⏳ **Language Detection Time:** {st.session_state.lang_detection_time} seconds")

    # 🌍 Translation Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Translate with Qwen"):
            st.session_state.selected_translation = "qwen"
    with col2:
        if st.button("🌍 Translate with DeepL"):
            st.session_state.selected_translation = "deepl"

# 🖼️ Display Uploaded/Selected Image
if st.session_state.image_uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(st.session_state.image_path, caption="Original Image", use_container_width=True)

        # ✅ Show Google Lens Image (for sample images only)
        if st.session_state.show_google_lens and selected_image in google_lens_images:
            st.subheader("📷 Google Lens Translation")
            st.image(f"google_lens_outputs/{google_lens_images[selected_image]}", caption="Google Lens Output", use_container_width=True)

# ✅ Translation & Replace Text in Image
if st.session_state.selected_translation:
    translated_texts = []

    start_time = time.time()

    for text, (x, y, w, h), _ in st.session_state.paragraphs:
        cropped_image = orig_image[y:y+h, x:x+w]
        detected_lang_code = ocr_translator.qwen(cropped_image)

        if detected_lang_code != "en":
            translated_text = (
                ocr_translator.qwen_translate_to_english(cropped_image, text)
                if st.session_state.selected_translation == "qwen"
                else ocr_translator.deepl_translate_to_english(text, detected_lang_code)
            )
        else:
            translated_text = text

        translated_texts.append(translated_text)

    # Replace text with translation in image
    output_image_path, _ = ocr_translator.replace_text_with_translation(
        st.session_state.image_path, st.session_state.paragraphs, translated_texts
    )

    # Load processed image
    processed_image = cv2.imread(output_image_path)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Store results in session state
    st.session_state.processed_image = processed_image

# ✅ Show Translated Image
if st.session_state.processed_image is not None:
    with col2:
        st.subheader("📷 Translated Image")
        st.image(st.session_state.processed_image, caption="Translated Image", use_container_width=True)
