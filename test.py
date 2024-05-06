import requests
import json
import base64
import numpy as np
import cv2
import matplotlib.pyplot as plt

url = "http://localhost:8004/prediction"

with open("images/123648868.jpg", "rb") as image_file:
    image_bytes = image_file.read()

files = [
    ("files", ("file1.jpg", open("images/123648868.jpg", "rb"), "image/jpeg")),
    ("files", ("file2.jpg", open("images/59359031.jpg", "rb"), "image/jpeg")),
]

headers = {'Content-Type': 'application/octet-stream'}
response = requests.post(url, files=files)

print(response.status_code)
print(response.text)
print(len(response.text))
