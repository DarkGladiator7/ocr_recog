import cv2
from .text_wrapping import wrap_text_to_fit

def replace_text_with_translation(image, paragraphs, translated_texts):
    """Replaces extracted paragraphs with translated text inside respective bounding boxes."""

    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width, _ = image.shape
    new_image = image.copy()  

    for i, (_, (x, y, w, h)) in enumerate(paragraphs):
        if i < len(translated_texts):
            text = translated_texts[i].strip()

            # Remove existing text
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(h / 50, 1.0))
            #font_scale = 0.5
            thickness = 2
            text_color = (0, 0, 0)

            # Wrap text to fit inside the bounding box
            wrapped_lines = wrap_text_to_fit(text, w + 10, font, font_scale, thickness)

            # Calculate text height
            total_text_height = sum(cv2.getTextSize(line, font, font_scale, thickness)[0][1] for line in wrapped_lines) + (len(wrapped_lines) - 1) * 10

            y_offset = y + (h - total_text_height) // 2 + 5

            for line in wrapped_lines:
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                text_x = x

                cv2.putText(new_image, line, (text_x, y_offset), font, font_scale, text_color, thickness)
                y_offset += text_size[1] + 8  

    return new_image
