import requests
import json
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt

url = "http://localhost:8004/prediction"


files = [
    ("files", ("59359031.jpg", open("images/59359031.jpg", "rb"), "image/jpeg")),
    ("files", ("123152360.jpg", open("images/123152360.jpg", "rb"), "image/jpeg")),
    ("files", ("123648868.jpg", open("images/123648868.jpg", "rb"), "image/jpeg")),
    ("files", ("125024830.jpg", open("images/125024830.jpg", "rb"), "image/jpeg")),
]


response = requests.post(url, files=files)

print(response.status_code)
print(response.text)
print(len(response.text))
