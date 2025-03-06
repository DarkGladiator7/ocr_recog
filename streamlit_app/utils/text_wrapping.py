import cv2

def calculate_font_scale_from_area(avg_char_area, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1):
    """Calculate the optimal font scale based on the average character area."""
    ref_text = "A"  # Use an uppercase letter as reference
    ref_size = cv2.getTextSize(ref_text, font, 1.0, thickness)[0]

    ref_char_width, ref_char_height = ref_size[0], ref_size[1]
    ref_char_area = ref_char_width * ref_char_height  # Area at scale 1.0

    # Calculate the optimal font scale using square root ratio
    font_scale = (avg_char_area / ref_char_area) ** 0.5

    return font_scale

def wrap_text(text, max_width, font_scale, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1):
    """Wrap text so it fits within the given max width."""
    words = text.split()
    wrapped_lines = []
    line = ""

    for word in words:
        test_line = line + " " + word if line else word
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        
        if text_size[0] <= max_width:
            line = test_line
        else:
            wrapped_lines.append(line)
            line = word

    if line:
        wrapped_lines.append(line)

    return wrapped_lines
