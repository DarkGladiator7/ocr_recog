import cv2
import numpy as np
from utils.text_wrapping import wrap_text, calculate_font_scale_from_area

def overlay_translated_text_on_image(paragraphs, translated_texts, original_image):
    """Overlay translated text on the original image at the same positions."""
    image_copy = original_image.copy()  # Preserve the original image

    for i, (_, (x, y, w, h), apc) in enumerate(paragraphs):
        image_copy[y: y+h, x: x+w] = (255, 255, 255)  # White background for translated text
        if i < len(translated_texts):
            text = translated_texts[i].strip()

            # Calculate the optimal font scale using character area
            font_scale = calculate_font_scale_from_area(avg_char_area=apc)

            image_copy = put_paragraph(image_copy, text, (x, y, w, h), font_scale)

    return image_copy
    
def put_paragraph(image, text, bbox, init_fontscale=1, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2, color=(0, 0, 0)):
    """Place wrapped text within a bounding box while ensuring it fits properly."""
    x, y, box_w, box_h = bbox
    font_scale = calculate_font_scale_from_area(avg_char_area=init_fontscale)

    wrapped_lines = wrap_text(text, box_w, font_scale, font, thickness)
    line_height = int(1.2 * cv2.getTextSize("A", font, font_scale, thickness)[0][1])
    
    cursor_y = y
    for line in wrapped_lines:
        text_size, baseline = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = x + (box_w - text_size[0]) // 2  # Center-align text
        text_y = cursor_y + text_size[1]

        if text_y > y + box_h:
            break  # Stop if text exceeds bounding box height

        cv2.putText(image, line, (text_x, text_y), font, font_scale, color, thickness)
        cursor_y += line_height + baseline

    return image
