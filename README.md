## Установка

``` cmd
git clone https://github.com/DarkPhoenix-777/wb-reputation.git
cd wb-reputation
docker compose up -d
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

API развернута на Yandex Cloud по адресу:  
<a href=158.160.169.93:8000/test>158.160.169.93:8000</a>  
(Там только CPU, поэтому может долго считать)

## Обучение

Для дообучения классификатора поместить в volume `data` в папку images изображения, и csv файл в формате: 

``` csv
image,label,subset
59359031.jpg,0,t
...
```
image - файл изображения  
label - метка(1 - фрод, 0 - не фрод)  
subset - выборка(t - обучающая, v - валидация)  

Далее отправить GET запрос:

http://host_name:8000/train

### Можно установить следующие параметры в запросе:

- n_epoch:  
    - Максимальное количество эпох
- early_stopping_rounds:  
    - Если не задан, то будет обучаться заданное количество эпох
    - Если int, то остановит обучение, если лосс на валидации не уменьшается переданное количество эпох
- learning_rate:  
    - Скорость обучения  
- log_batch:  
    - Если True - выводит лосс на обучении по батчу в консоль