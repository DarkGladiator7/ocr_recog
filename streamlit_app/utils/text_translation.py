import deepl
from langdetect import detect
import os 
from dotenv import load_dotenv
load_dotenv()
 
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
translator = deepl.Translator(DEEPL_API_KEY)

def translate_text(text):
    """Translate non-English text to English using DeepL."""
    detected_lang = detect(text).upper()
    
    if detected_lang != "EN":  # Translate only if not already English
        try:
            result = translator.translate_text(text, source_lang=detected_lang, target_lang="EN-US")
            return result.text
        except deepl.DeepLException as e:
            print(f"DeepL Error: {e}")
            return text  # Return original if translation fails
    return text
