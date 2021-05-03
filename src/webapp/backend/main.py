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

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.get("/getModel")
def get_model():
    return {"message": "MODEL"}

@app.post("/addData")
def add_data(file: UploadFile = File(...)):
    return {"message": file.filename}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
