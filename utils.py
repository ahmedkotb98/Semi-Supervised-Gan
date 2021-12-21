import io
import json
import random

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from Nets import Discriminator, Generator

clasess = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_path):
    model = torch.load(model_path)
    model.eval()

    return model


def transform_image(image_bytes): 
    
    transform_image = transforms.Compose([
        
        transforms.Resize((28,28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    return transform_image(image).unsqueeze(0)


def get_prediction(image, model):
    im = transform_image(image)
    with torch.no_grad():
        ret = torch.max(model(im), 1)[1].data
            
    return clasess[np.array(ret)[0]]


'''
FROM python:3.8.12-slim

WORKDIR /usr/home

COPY ./requirements.txt .
 
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
'''
