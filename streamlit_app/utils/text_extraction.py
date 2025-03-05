import pytesseract

def extract_paragraphs_with_bounding_boxes(image):
    """Extract paragraphs and their bounding boxes."""
    custom_config = r'--oem 3 --psm 3'
    data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

    paragraphs = []
    paragraph_text = []
    paragraph_boxes = []
    prev_y = None

    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            if prev_y is not None and (y - prev_y) > h * 3:
                if paragraph_text:
                    x_min = min([b[0] for b in paragraph_boxes])
                    y_min = min([b[1] for b in paragraph_boxes])
                    x_max = max([b[0] + b[2] for b in paragraph_boxes])
                    y_max = max([b[1] + b[3] for b in paragraph_boxes])
                    paragraphs.append((" ".join(paragraph_text), (x_min, y_min, x_max - x_min, y_max - y_min)))

                paragraph_text = []  
                paragraph_boxes = []

            paragraph_text.append(text)
            paragraph_boxes.append((x, y, w, h))
            prev_y = y

    if paragraph_text:
        x_min = min([b[0] for b in paragraph_boxes])
        y_min = min([b[1] for b in paragraph_boxes])
        x_max = max([b[0] + b[2] for b in paragraph_boxes])
        y_max = max([b[1] + b[3] for b in paragraph_boxes])
        paragraphs.append((" ".join(paragraph_text), (x_min, y_min, x_max - x_min, y_max - y_min)))

    return paragraphs
