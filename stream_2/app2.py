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

# Title
st.title("📄 OCR Image Translator")

# Upload Image Button (Top)
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# Reset session state when a new image is uploaded
if uploaded_file is not None:
    st.session_state.image_uploaded = False
    st.session_state.processed_image = None
    st.session_state.detected_languages.clear()
    st.session_state.translated_data.clear()
    st.session_state.paragraphs.clear()
    st.session_state.lang_detection_time = None  
    st.session_state.translation_time = None  

    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_path = "uploaded_image.jpg"
    cv2.imwrite(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

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
        detected_lang = ocr_translator.qwen(cropped_image)  
        detected_languages.add(detected_lang)

    # **End timer for language detection**
    st.session_state.lang_detection_time = round(time.time() - start_time, 2)

    # Update session state
    st.session_state.image_path = image_path
    st.session_state.image_uploaded = True
    st.session_state.detected_languages = detected_languages
    st.session_state.paragraphs = paragraphs

# Show detected language **only after an image is uploaded**
if st.session_state.image_uploaded:
    st.info(f"**Detected Language(s):** {', '.join(st.session_state.detected_languages) if st.session_state.detected_languages else 'N/A'}")
    
    if st.session_state.lang_detection_time is not None:
        st.write(f"⏳ **Language Detection Time:** {st.session_state.lang_detection_time} seconds")

# **Translation Buttons**
translate_qwen = False
translate_google = False
if st.session_state.image_uploaded:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Translate with Qwen"):
            translate_qwen = True
            # Reset previous output
            st.session_state.processed_image = None
            st.session_state.translated_data = []
            st.session_state.translation_time = None
    with col2:
        if st.button("🌍 Translate with Google Translate"):
            translate_google = True
            # Reset previous output
            st.session_state.processed_image = None
            st.session_state.translated_data = []
            st.session_state.translation_time = None

# Display uploaded image (Left Side)
if st.session_state.image_uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(st.session_state.image_path, caption="Original Image", use_container_width=True)

    # Process Translation ONLY if a button is clicked
    if translate_qwen or translate_google:
        translated_data = []
        translated_texts = []

        # **Start timer for translation**
        start_time = time.time()

        for text, (x, y, w, h), _ in st.session_state.paragraphs:
            cropped_image = orig_image[y:y+h, x:x+w]  
            detected_lang = ocr_translator.qwen(cropped_image)  

            if detected_lang != "en":
                if translate_qwen:
                    translated_text = ocr_translator.qwen_translate_to_english(cropped_image, text)
                elif translate_google:
                    translated_text = ocr_translator.deepl_translate_to_english(text)
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

    # **Download Processed Image**
    processed_image_pil = Image.fromarray(st.session_state.processed_image)
    img_buffer = io.BytesIO()
    processed_image_pil.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    st.download_button(
        label="📥 Download Translated Image",
        data=img_bytes,
        file_name="translated_image.png",
        mime="image/png"
    )

    # **Download JSON Output**
    json_data = json.dumps(st.session_state.translated_data, indent=4, ensure_ascii=False).encode("utf-8")
    st.download_button(
        label="📥 Download JSON",
        data=json_data,
        file_name="translated_text.json",
        mime="application/json"
    )
