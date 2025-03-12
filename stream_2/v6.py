import statistics
from segmentation1.paragraph_detector import extract_paragraphs
import cv2
import pytesseract
import numpy as np
import json
import base64
import requests


class VisionLLM:
    def __init__(self, image_path):
        self.image_path = image_path
        self.tesseract_lang_map = {
            "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita",
            "zh-cn": "chi_sim", "zh-tw": "chi_tra", "ja": "jpn", "ko": "kor",
            "hi": "hin", "ar": "ara", "ru": "rus", "pt": "por", "bn": "ben",
            "ta": "tam", "te": "tel", "ml": "mal"
        }

    def preprocess_image(self, image_path):
        """Preprocess the image for better OCR accuracy."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image from path: {image_path}")
            return None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        print(f"Preprocessed image size: {image.shape}")
        return image, binary

    def find_apc(self, p_boxes, p_texts):
        apcs = []
        for bbox, text in zip(p_boxes, p_texts):
            w, h = bbox[2], bbox[3]
            area = w * h
            text_len = len(text)
            avg_apc = area / text_len
            apcs.append(avg_apc)
        avg_apc = statistics.mean(apcs)
        return avg_apc


    def extract_paragraphs_with_bounding_boxes(self, image):
        """Extract paragraphs and their bounding boxes."""
        custom_config = r'--oem 3 --psm 3'
        data = pytesseract.image_to_data(
            image, config=custom_config, output_type=pytesseract.Output.DICT)

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
                        apc = self.find_apc(
                            p_boxes=paragraph_boxes, p_texts=paragraph_text)
                        paragraphs.append(
                            (" ".join(paragraph_text), (x_min, y_min, x_max - x_min, y_max - y_min), apc))

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
            apc = self.find_apc(p_boxes=paragraph_boxes,
                                p_texts=paragraph_text)
            paragraphs.append(
                (" ".join(paragraph_text), (x_min, y_min, x_max - x_min, y_max - y_min), apc))

        return paragraphs

    def extract_text(self, image, detected_lang_codes=None):
        """Extract text from the image using Tesseract OCR with multiple language support."""
        custom_config = r'--oem 3 --psm 3'

        if not isinstance(image, np.ndarray):
            image = np.array(image)  # Convert PIL image to OpenCV format if needed

        if detected_lang_codes:
            print(f"Detected Languages: {detected_lang_codes}")
            lang_list = [self.tesseract_lang_map.get(lang, 'eng') for lang in detected_lang_codes]
            combined_langs = "+".join(set(lang_list))  # Combine languages for Tesseract
            custom_config += f' -l {combined_langs}'
        else:
            print("No Detected Language Code provided. Using default: 'eng'")
            custom_config += ' -l eng'

        # Perform OCR using pytesseract
        print(f'CONFIG:\n{custom_config}')
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

        text_bboxes = []
        full_text = ""

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                text_bboxes.append({'word': text, 'b_box': (x, y, x + w, y + h)})
                full_text += text + " "

        print("Extracted Text: ", full_text)
        return text_bboxes


    def deepl_translate_to_english(self, text, source_lang1):
        url = "https://dl.localzoho.com/api/v2/nlp/translation/translate"

        payload = json.dumps({
            "text": {
                "sentences": [text]
            },
            "source_language": source_lang1,
            "target_language": "en",
            "align": False
        })

        headers = {
            'Authorization': 'Zoho-oauthtoken 1000.a7a474d42e9ba89d77501ebe738c541b.986be26b6cdd6746d07af53f605d9fa4',
            'Content-Type': 'application/json'
        }

        try:
            # Sending the request to Deepl API
            print(f"Sending to Deepl API: {payload}")  # Debug print
            response = requests.post(url, headers=headers, data=payload)

            # Debug prints to check the response status and content
            print(f"Response Status Code: {response.status_code}")  # Debug print
            print(f"Response Content: {response.text}")  # Debug print

            # Check the content type
            print(f"Content-Type: {response.headers.get('Content-Type')}")  # Check the content type of the response

            # Only continue if the response is a success (status code 200)
            if response.status_code == 200:
                try:
                    # Parse the JSON response
                    response_data = response.json()
                    print(f"Parsed JSON Response: {response_data}")  # Debug print

                    # Check for 'translations' key in the response
                    if 'translation' in response_data:
                        translations = response_data['translation']
                        if len(translations) > 0:
                            # Extract the translated text
                            translation = translations[0].get('translate', '')
                            print(f"Translated Text: {translation}")  # Debug print
                            return translation
                        else:
                            print("Error: 'translations' exists but is empty.")
                            return "Translation Error1"
                    else:
                        print("Error: 'translations' key not found in the response.")
                        return "Translation Error2"
                except ValueError as e:
                    # Error parsing JSON response
                    print(f"Error parsing JSON response: {e}")
                    return "Translation Error3"
            else:
                print(f"Error: Received non-200 status code from Deepl API: {response.status_code}")
                return "Translation Error4"

        except Exception as e:
            # Handle any exceptions during the request
            print(f"Error occurred while contacting Deepl API: {str(e)}")
            return "Translation Error5"

    def qwen(self, cropped, user_prompt="Identify the language of the text in this image.",
             sys_prompt="Return only the language code like 'en' for English, 'fr' for French.") -> str:
        """Detect the language of text in an image using the qwen API."""
        result = None
        try:
            encoded_image = cv2.imencode('.jpg', cropped)[1]
            img1 = base64.b64encode(encoded_image.tobytes()).decode()

            url = "http://crmgpu5-10042.csez.zohocorpin.com:8781/qwen/inference"
            headers = {"Content-Type": "application/json"}

            payload = json.dumps({
                "prompt": user_prompt,
                "temperature": 0.000000001,
                "images": [img1],
                "system_prompt": sys_prompt
            })

            response = requests.post(url, headers=headers, data=payload)

            if response.status_code == 200:
                result = response.json().get('response', '')
                print(f"API response: {result}")
            else:
                print(f"Received status code {response.status_code} from API")
        except Exception as e:
            print(f"Error during API call: {e}")
        return result

    def qwen_translate_to_english(self, cropped, text, user_prompt="Translate this text to English.",
                                  sys_prompt="Return only the translated text.") -> str:
        """Translate extracted text to English using the qwen API, including the image."""
        result = None
        try:
            encoded_image = cv2.imencode('.jpg', cropped)[1]
            img1 = base64.b64encode(encoded_image.tobytes()).decode()

            url = "http://crmgpu5-10042.csez.zohocorpin.com:8781/qwen/inference"
            headers = {"Content-Type": "application/json"}

            payload = json.dumps({
                "prompt": user_prompt,
                "temperature": 0.000000001,
                "text": text,
                "images": [img1],
                "system_prompt": sys_prompt
            })

            response = requests.post(url, headers=headers, data=payload)

            if response.status_code == 200:
                result = response.json().get('response', '')
                print(f"Translation result: {result}")
            else:
                print(f"Received status code {response.status_code} from API")
        except Exception as e:
            print(f"Error during API call: {e}")

        return result

    def overlay_translated_text_on_image(self, paragraphs, translated_texts, original_image):
        """Overlay translated text on the original image at the same positions."""
        image_copy = original_image.copy()  # Preserve the original image

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        for i, (_, (x, y, w, h), apc) in enumerate(paragraphs):
            image_copy[y: y + h, x: x + w] = (255, 255, 255)
            if i < len(translated_texts):
                text = translated_texts[i].strip()

                font_scale = self.calculate_font_scale_from_area(
                    avg_char_area=apc)

                image_copy = self.put_paragraph(img=image_copy,
                                                text=text,
                                                bbox=(x, y, w, h),
                                                init_fontscale=font_scale)

        return image_copy

    def detect_and_translate_text(self, paragraphs, original_image):
        """Detects language and translates only non-English text."""
        translated_texts = []
        for text, (x, y, w, h), _ in paragraphs:
            # Crop the image based on the bounding box
            cropped_image = original_image[y:y + h, x:x + w]
            # Detect the language using qwen
            detected_lang = self.qwen(cropped_image)  # Detects the language here
            print(f"Detected language: {detected_lang}")

            # Now pass the detected language to the `extract_text` function
            text_bboxes = self.extract_text(cropped_image, detected_lang_code=detected_lang)

            # Proceed with translation logic if the text is not in English
            if detected_lang != "en":
                translated_text = self.qwen_translate_to_english(cropped_image, text)  # Translate using qwen
                translated_texts.append(translated_text)
            else:
                translated_texts.append(text)

        return translated_texts

    def replace_text_with_translation(self, image_path, paragraphs, translated_texts):
        """Replaces extracted paragraphs with translated text inside respective bounding boxes."""
        original = cv2.imread(image_path)
        if original is None:
            print(f"Error: The image at {image_path} could not be loaded.")
            return None, None

        # Overlay translated text
        output_image = self.overlay_translated_text_on_image(
            paragraphs, translated_texts, original)

        # Save the translated image
        output_image_path = "anas2.jpg"
        cv2.imwrite(output_image_path, output_image)
        print(f"Translated image saved at: {output_image_path}")

        # Save changes as a JSON file
        changes_json_path = "changes.json"
        changes = [{"original": paragraphs[i][0], "translated": translated_texts[i]}
                   for i in range(len(translated_texts))]
        with open(changes_json_path, "w", encoding="utf-8") as f:
            json.dump(changes, f, indent=4, ensure_ascii=False)
        print(f"Changes saved to: {changes_json_path}")

        return output_image_path, changes_json_path

    def calculate_font_scale_from_area(self, avg_char_area, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1):
        # Get reference size at scale 1.0 using a single character
        ref_text = "A"  # Use an uppercase letter as reference
        ref_size = cv2.getTextSize(ref_text, font, 1.0, thickness)[0]
        ref_char_width, ref_char_height = ref_size[0], ref_size[1]
        ref_char_area = ref_char_width * ref_char_height  # Area at scale 1.0

        # Calculate the optimal font scale using square root ratio
        font_scale = (avg_char_area / ref_char_area) ** 0.5

        return font_scale

    def get_optimal_font_scale(self, text, bbox, font_scale=1.0, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=1):

        box_w, box_h = bbox

        while font_scale > 0:
            # Temporary image to test text fitting
            img = np.zeros((box_h, box_w, 3), dtype=np.uint8)
            wrapped_lines = self.wrap_text(
                text, box_w, font, font_scale, thickness)
            total_height = sum(
                cv2.getTextSize(line, font, font_scale, thickness)[0][1] + 5 for line in wrapped_lines)  # Add spacing

            if total_height <= box_h:
                return font_scale  # Found suitable font scale

            font_scale -= 0.05  # Reduce font scale

        return 0.1  # Minimum font scale to prevent zero scaling

    def wrap_text(self, text, box_w, font, font_scale, thickness):

        words = text.split()
        lines = []
        line = ""

        for word in words:
            test_line = line + " " + word if line else word
            text_size, _ = cv2.getTextSize(
                test_line, font, font_scale, thickness)

            if text_size[0] <= box_w:
                line = test_line  # Add word to current line
            else:
                lines.append(line)  # Store full line
                line = word  # Start new line

        if line:
            lines.append(line)  # Append the last line

        return lines

    def put_paragraph(self, img, text, bbox, init_fontscale=1, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):

        x, y, box_w, box_h = bbox
        font_scale = self.get_optimal_font_scale(
            text, (box_w, box_h), init_fontscale, font, thickness)  # Find best font scale
        font_scale = font_scale * 0.75
        lines = self.wrap_text(text, box_w, font, font_scale, thickness)

        cursor_y = y
        for line in lines:
            text_size, baseline = cv2.getTextSize(
                line, font, font_scale, thickness)
            text_height = text_size[1]

            # Stop if exceeding bounding box height
            if cursor_y + text_height > y + box_h:
                break

            cv2.putText(img, line, (x, cursor_y + text_height),
                        font, font_scale, (0, 0, 0), thickness)
            cursor_y += text_height + baseline + 5  # Move cursor down with spacing

        return img


# def draw_boxes(image, paragraphs_info):
#     from PIL import Image
#     boxes = [paragraph_info[1] for paragraph_info in paragraphs_info]
#     for x1, y1, w, h in boxes:
#         image = cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
#     Image.fromarray(image).show()


# Example usage:
def main():
    image_path = r"E:\Data\Downloads\ocr_recog-main\stream_2\sample_images\par.jpeg"
    ocr_translator = VisionLLM(image_path)

    image, binary = ocr_translator.preprocess_image(image_path)

    # Extract text from image
    text_bboxes = ocr_translator.extract_text(image)  # Initial text extraction to get bounding boxes

    # Extract paragraphs with bounding boxes
    paragraphs = extract_paragraphs(image, texts_b_boxes=text_bboxes)

    # Detect language and translate non-English text
    translated_texts = ocr_translator.detect_and_translate_text(paragraphs, image)

    # Replace text with translation in the image
    output_image, changes_json = ocr_translator.replace_text_with_translation(image_path, paragraphs, translated_texts)


if __name__ == '__main__':
    # Run the main function
    main()
