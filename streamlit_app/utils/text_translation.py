from googletrans import Translator
from langdetect import detect

translator = Translator()

def translate_text(text):
    """Translate non-English text to English using Google Translate."""
    detected_lang = detect(text).upper()
    

    if detected_lang != "EN":  # Translate only if not already English
        try:
            result = translator.translate(text, src=detected_lang, dest="en")
            return result.text
        except Exception as e:
            print(f"Google Translate Error: {e}")
            return text  # Return original if translation fails
            