import cv2
import base64
import json
import requests

def qwen_translate_to_english(cropped_image, text, user_prompt="Translate this text to English.", sys_prompt="Return only the translated text."):
    """Translate extracted text to English using the Qwen API, including the cropped image."""
    result = None
    success = False  

    try:
        url = "http://crmgpu5-10042.csez.zohocorpin.com:8781/qwen/inference"
        headers = {"Content-Type": "application/json"}

        # Encode Image to Base64
        _, buffer = cv2.imencode('.jpg', cropped_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "prompt": user_prompt,
            "temperature": 0.000000001,
            "text": text,
            "images": [image_base64],  # 🔥 Fix - Send image as List
            "system_prompt": sys_prompt
        }

        # Debug Payload
        print("Payload:", json.dumps(payload))

        # Send Request
        response = requests.post(url, headers=headers, json=payload)
        print("Response Status:", response.status_code)
        print("Response Text:", response.text)

        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            success = bool(result)
        else:
            print(f"Qwen API Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error during Qwen API call: {e}")

    return result if success else text
