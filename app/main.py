import os
import sys
import logging
import pathlib

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

from utils import (
    set_seed,
    load_model,
    get_prediction
)

app = FastAPI()

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent

model_file_path = os.path.join(ROOT_PATH, "models", "D.pkl")
model = load_model(model_file_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):    
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')    

    try:
        contents = await file.read()

        predicted_class = get_prediction(contents, model)
        
        logging.info(f"Predicted Class: {predicted_class}")

        return {            
            "class": predicted_class,
            "status_code": 200
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))