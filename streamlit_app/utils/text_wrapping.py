import cv2

def wrap_text_to_fit(text, max_width, font, font_scale, thickness):
    """Splits text into multiple lines so it fits within a given width."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        temp_line = current_line + " " + word if current_line else word
        text_size = cv2.getTextSize(temp_line, font, font_scale, thickness)[0]

        if text_size[0] <= max_width:
            current_line = temp_line  
        else:
            lines.append(current_line)  
            current_line = word  

    if current_line:
        lines.append(current_line)

    return lines
