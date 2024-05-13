## Установка

``` cmd
git clone https://github.com/DarkPhoenix-777/wb-reputation.git
cd wb-reputation
docker compose up
```

## Использование
``` python
import requests
import io
from PIL import Image

def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

url = "http://localhost:8000/"


# Отправка одного файла
file = image_to_byte_array(Image.open("img1.jpg"))
headers = {'Content-Type': 'application/octet-stream'}

response = requests.post(url + "prediction", data=file, headers=headers)

response_data = response.json()

print(response_data["prob"], response_data["verdict"])

# Отправка нескольких файлов
files = [
    ("files", ("img1", image_to_byte_array(Image.open("img1.jpg")))),
    ("files", ("img2", image_to_byte_array(Image.open("img2.jpg")))),
    ("files", ("img3", image_to_byte_array(Image.open("img3.jpg")))),
    ("files", ("img4", image_to_byte_array(Image.open("img1.jpg")))),
]

response = requests.post(url + "prediction_batch", files=files)

for data in response.json():
    print(data["image"], data["prob"], data["verdict"])

```