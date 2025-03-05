import requests
import json

def qwen_translate_to_english(text, user_prompt="Translate this text to English.", sys_prompt="Return only the translated text."):
    """Translate extracted text to English using the Qwen API (without image input)."""
    result = None
    success = False  # Track if translation was successful

    try:
        # Qwen API endpoint
        url = "http://crmgpu5-10042.csez.zohocorpin.com:8781/qwen/inference"
        headers = {"Content-Type": "application/json"}

        # Prepare payload for translation
        payload = json.dumps({
            "prompt": user_prompt,
            "temperature": 0.000000001,
            "text": text,  # Only the text is sent for translation
            "system_prompt": sys_prompt
        })

        # Send request to Qwen API
        response = requests.post(url, headers=headers, data=payload)

        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            success = True if result else False  # Check if translation was successful
        else:
            print(f"Qwen API Error: Received status code {response.status_code}")

    except Exception as e:
        print(f"Error during Qwen API call: {e}")

    return result, success
