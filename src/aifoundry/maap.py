import json
import os

import requests


def generate_medical_image_embeddings(encoded_image: str, text_descr: str) -> dict:
    data = {
        "input_data": {
            "columns": ["image", "text"],
            "index": [0],
            "data": [
                [encoded_image, text_descr]
            ],
        },
        "params": {"get_scaling_factor": True},
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('MI2_MODEL_KEY')}"
    }

    url = os.environ.get("MI2_MODEL_ENDPOINT")
    response = requests.post(url, headers=headers, json=data)
    return json.loads(response.text)[0]

def generate_nonclinical_radiology_report(encoded_image: str, text_descr: str) -> dict:
    data = {
        "input_data": {
            "data": [
                [
                    encoded_image,
                    text_descr
                ]
            ],
            "columns": [
                "frontal_image",
                "indication"
            ],
            "index": [
                0
            ]
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('CXR_MODEL_KEY')}"
    }

    url = os.environ.get("CXR_MODEL_ENDPOINT")
    response = requests.post(url, headers=headers, json=data)
    return json.loads(response.text)[0]