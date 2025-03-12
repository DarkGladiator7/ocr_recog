import streamlit as st
import cv2
import numpy as np
import time
from vision_llm import VisionLLM  # Importing your class
from PIL import Image
from segmentation.paragraph_detector import extract_paragraphs

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="OCR Image Translator")

# Language Mapping
LANGUAGE_MAPPING = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
    "ar": "Arabic", "ru": "Russian", "it": "Italian", "pt": "Portuguese",
    "tr": "Turkish", "nl": "Dutch", "sv": "Swedish", "hu": "Hungarian"
}

# Sample images
sample_images = ["hin.jpg", "jap.png", 'par.jpeg', 'fren.jpg', 'araa.png']

# Google Lens Translated Images
google_lens_images = {
    "hin.jpg": "lens_hin.jpg",
    "jap.png": "lens_jap.JPG",
    "par.jpeg": "lens_par.JPG",
    "fren.jpg": "lens_fre.JPG",
    "araa.png": "lens_arab.JPG"
}

# ✅ Initialize session state
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
if "paragraphs" not in st.session_state:
    st.session_state.paragraphs = []
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "show_google_lens" not in st.session_state:
    st.session_state.show_google_lens = False

# Title
st.title("📄 OCR Image Translator")

# Input Method Selection
input_method = st.radio("Choose Input Method", ["Select from Sample", "Upload Image"])

# 🔄 Reset session state when a new image is chosen
if st.session_state.input_mode != input_method:
    st.session_state.image_uploaded = False
    st.session_state.selected_translation = None  # 🔥 Reset translation choice
    st.session_state.processed_image = None
    st.session_state.detected_languages = set()
    st.session_state.paragraphs = []
    st.session_state.image_path = None
    st.session_state.show_google_lens = False
    st.session_state.input_mode = input_method  # Update input mode

# 🎯 Image Selection (Dropdown)
if input_method == "Select from Sample":
    selected_image = st.selectbox("Select an Image", ["Select an image"] + sample_images)

    if selected_image != "Select an image":
        st.session_state.image_path = f"sample_images/{selected_image}"
        st.session_state.show_google_lens = True
        st.session_state.image_uploaded = True
        st.session_state.selected_translation = None  # 🔥 Ensure translation waits for user input

# 📤 Image Upload
elif input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Save uploaded image
        st.session_state.image_path = "uploaded_image.jpg"
        cv2.imwrite(st.session_state.image_path, image_bgr)

        st.session_state.show_google_lens = False
        st.session_state.image_uploaded = True
        st.session_state.selected_translation = None  # 🔥 Reset translation on new upload

# 🖼️ Process Image (only if image is selected/uploaded)
if st.session_state.image_uploaded and st.session_state.image_path:
    image = Image.open(st.session_state.image_path)
    ocr_translator = VisionLLM(st.session_state.image_path)

    # Preprocess Image
    orig_image, _ = ocr_translator.preprocess_image(st.session_state.image_path)

    # Extract paragraphs
    text_bboxes = ocr_translator.extract_text(image)
    paragraphs = extract_paragraphs(image, texts_b_boxes=text_bboxes)

    # Detect language
    detected_languages = set()
    for text, (x, y, w, h), _ in paragraphs:
        cropped_image = orig_image[y:y+h, x:x+w]
        detected_lang_code = ocr_translator.qwen(cropped_image)
        detected_lang_name = LANGUAGE_MAPPING.get(detected_lang_code, detected_lang_code)
        detected_languages.add(detected_lang_name)

    # Update session state
    st.session_state.detected_languages = detected_languages
    st.session_state.paragraphs = paragraphs

# 🔍 Show Detected Language
if st.session_state.image_uploaded:
    st.info(f"**Detected Language(s):** {', '.join(st.session_state.detected_languages) if st.session_state.detected_languages else 'N/A'}")

    # 🌍 Translation Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Translate with Qwen"):
            st.session_state.selected_translation = "qwen"
    with col2:
        if st.button("🌍 Translate with Zoho Translator"):
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

# ✅ Translation & Replace Text in Image (ONLY after user clicks button)
if st.session_state.selected_translation:
    translated_texts = []

    for text, (x, y, w, h), _ in st.session_state.paragraphs:
        cropped_image = orig_image[y:y+h, x:x+w]
        detected_lang_code = ocr_translator.qwen(cropped_image)

        if detected_lang_code != "en":
            translated_text = (
                ocr_translator.qwen_translate_to_english(cropped_image, text)
                if st.session_state.selected_translation == "qwen"
                else ocr_translator.zoho_translate_to_english(text, detected_lang_code)
            )
        else:
            translated_text = text

        translated_texts.append(translated_text)
        
        
    # Replace text in image
    output_image_path, _ = ocr_translator.replace_text_with_translation(
        st.session_state.image_path, st.session_state.paragraphs, translated_texts
    )

    # Load processed image
    processed_image = cv2.imread(output_image_path)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Store in session state
    st.session_state.processed_image = processed_image

# ✅ Show Translated Image
if st.session_state.processed_image is not None:
    with col2:
        st.subheader("📷 Translated Image")
        st.image(st.session_state.processed_image, caption="Translated Image", use_container_width=True)
