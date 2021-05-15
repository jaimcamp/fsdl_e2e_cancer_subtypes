#!/usr/bin/env python
"""
File: main.py
Author: yourname
Email: yourname@email.com
Github: https://github.com/yourname
Description:
"""

from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
from starlette.responses import PlainTextResponse
from predict_model import updateTSNE

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.get("/getModel")
def get_model():
    return {"message": "MODEL"}

@app.post("/addData")
def add_data(file: UploadFile = File(...)):
    tsne_path = updateTSNE(modelpath='/storage/model_save.pth',
                         newdatapath=file.file.read(),
                         originaldatapath='/storage/tsne_df.parquet')
    return {"message": tsne_path}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
