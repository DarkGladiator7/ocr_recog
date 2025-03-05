import easyocr

# Initialize EasyOCR Reader with multiple languages
reader = easyocr.Reader(['en', 'de', 'fr', 'es'])  # English, German, French, Spanish

def extract_paragraphs_with_bounding_boxes(image):
    """Extract paragraphs and their bounding boxes using EasyOCR."""
    results = reader.readtext(image)  # Perform OCR

    paragraphs = []
    paragraph_text = []
    paragraph_boxes = []
    prev_y = None

    for (bbox, text, _) in results:
        (x_min, y_min), (x_max, y_max) = bbox[0], bbox[2]  # Get bounding box
        w, h = x_max - x_min, y_max - y_min

        if prev_y is not None and (y_min - prev_y) > h * 3:
            if paragraph_text:
                x_min_p = min([b[0] for b in paragraph_boxes])
                y_min_p = min([b[1] for b in paragraph_boxes])
                x_max_p = max([b[0] + b[2] for b in paragraph_boxes])
                y_max_p = max([b[1] + b[3] for b in paragraph_boxes])
                paragraphs.append((" ".join(paragraph_text), (x_min_p, y_min_p, x_max_p - x_min_p, y_max_p - y_min_p)))

            paragraph_text = []  
            paragraph_boxes = []

        paragraph_text.append(text)
        paragraph_boxes.append((x_min, y_min, w, h))
        prev_y = y_min

    if paragraph_text:
        x_min_p = min([b[0] for b in paragraph_boxes])
        y_min_p = min([b[1] for b in paragraph_boxes])
        x_max_p = max([b[0] + b[2] for b in paragraph_boxes])
        y_max_p = max([b[1] + b[3] for b in paragraph_boxes])
        paragraphs.append((" ".join(paragraph_text), (x_min_p, y_min_p, x_max_p - x_min_p, y_max_p - y_min_p)))

    return paragraphs
