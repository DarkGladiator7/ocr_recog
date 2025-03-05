from langdetect import detect

def detect_language(text):
    """Detect the language of the given text."""
    return detect(text)
