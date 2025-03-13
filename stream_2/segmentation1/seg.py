import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import easyocr

def get_page_wise_texts(images):
    reader = easyocr.Reader(lang_list=['en'])
    page_wise_texts = {}
    for pg_no, img in enumerate(images):
        results = reader.readtext(np.asarray(img))
        word_bbox_pairs = []
        for result in results:
            word = result[1]
            box = result[0]
            bbox = [box[0][0], box[0][1], box[2][0], box[2][1]]
            word_bbox_pairs.append({'word': word, 'b_box': bbox})
        page_wise_texts[pg_no] = word_bbox_pairs
    return page_wise_texts

def draw_bounding_boxes(image, bboxes, token_words):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Add label above the box
        text = token_words[i]
        draw.text((x1, y1 - 10), text, fill="red", font=font)

def main():
    FILE_NAME = "res.pdf"
    OUTPUT_FILE = "output_res.pdf"

    # ✅ Step 1: Convert PDF to images
    images = convert_from_path(FILE_NAME, use_cropbox=True)

    # ✅ Step 2: Extract text and bounding boxes
    page_wise_texts = get_page_wise_texts(images)

    processed_pages = []
    for page_no, img in enumerate(images):
        print(f"Processing Page {page_no + 1}...")

        bboxes = []
        token_words = []
        for text_info in page_wise_texts.get(page_no, []):
            bboxes.append(text_info['b_box'])
            token_words.append(text_info['word'])

        # ✅ Step 3: Draw bounding boxes directly onto the image
        draw_bounding_boxes(img, bboxes, token_words)

        # ✅ Step 4: Add to processed pages
        processed_pages.append(img)

    # ✅ Step 5: Save as a single PDF
    processed_pages[0].save(
        OUTPUT_FILE, save_all=True, append_images=processed_pages[1:]
    )

    print(f"✅ Processed PDF saved as '{OUTPUT_FILE}'")

if __name__ == '__main__':
    main()
