import requests
import json
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt

url = "http://localhost:8004/prediction"


files = [
    ("files", ("123648868.jpg", open("images/123648868.jpg", "rb"), "image/jpeg")),
    ("files", ("59359031.jpg", open("images/59359031.jpg", "rb"), "image/jpeg")),
]

headers = {'Content-Type': 'application/octet-stream'}
response = requests.post(url, files=files)

print(response.status_code)
print(response.text)
print(len(response.text))
