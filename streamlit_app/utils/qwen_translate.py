import json
import requests
import cv2
import base64
import numpy as np

def qwen_translate_to_english(cropped_image, text, user_prompt="Translate this text to English.", sys_prompt="Return only the translated text."):
    """Translate extracted text to English using the Qwen API (with image input)."""
    result = None
    success = False  

    try:
        url = "http://crmgpu5-10042.csez.zohocorpin.com:8781/qwen/inference"
        headers = {"Content-Type": "application/json"}

        # Encode the image in Base64
        _, buffer = cv2.imencode('.jpg', cropped_image)
        img_base64 = base64.b64encode(buffer).decode()

        # Prepare API payload with text and image
        payload = json.dumps({
            "prompt": user_prompt,
            "text": text,
            "images": [img_base64],  # Send the image
            "system_prompt": sys_prompt
        })

        # Send request to Qwen API
        response = requests.post(url, headers=headers, data=payload)

        print("Response Status:", response.status_code)
        print("Response Text:", response.text)  # Debugging: Print full response

        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            success = bool(result)  
        else:
            print(f"Qwen API Error: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error during Qwen API call: {e}")

    return result, success
