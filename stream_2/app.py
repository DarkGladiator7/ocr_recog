import streamlit as st
import cv2
import numpy as np
import json
import os
from PIL import Image
from vision_llm import VisionLLM  # Importing your class

# Streamlit UI
st.title("OCR Image Translator")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Save temporarily for processing
    temp_image_path = "temp_image.jpg"
    cv2.imwrite(temp_image_path, image)

    # Display original image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Initialize VisionLLM
    ocr_translator = VisionLLM(temp_image_path)

    # Process image
    with st.spinner("Processing image..."):
        image, binary = ocr_translator.preprocess_image(temp_image_path)
        paragraphs = ocr_translator.extract_paragraphs_with_bounding_boxes(image)

        # Detect and translate text
        translated_texts = ocr_translator.detect_and_translate_text(paragraphs, image)

        # Replace text with translation
        output_image_path, changes_json_path = ocr_translator.replace_text_with_translation(temp_image_path, paragraphs, translated_texts)

    # Display processed image
    output_image = Image.open(output_image_path)
    st.image(output_image, caption="Translated Image", use_column_width=True)

    # Show changes
    with open(changes_json_path, "r", encoding="utf-8") as f:
        changes = json.load(f)
    st.write("### Translation Changes")
    st.json(changes)

    # Provide download option for translated image
    with open(output_image_path, "rb") as file:
        st.download_button("Download Translated Image", file, file_name="translated_image.jpg", mime="image/jpeg")

    # Provide download option for JSON changes
    with open(changes_json_path, "rb") as file:
        st.download_button("Download Changes JSON", file, file_name="translation_changes.json", mime="application/json")

    # Clean up temporary files
    os.remove(temp_image_path)
    os.remove(output_image_path)
    os.remove(changes_json_path)
