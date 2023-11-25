from app.photo import photo
from flask import request
import requests
from flask import current_app as app, make_response, jsonify, send_file
from flask_cors import cross_origin
import base64
import csv
import io
import os
import numpy as np
from PIL import Image
import torch


@cross_origin()
@photo.post("/api/photo")
def make_correction():
    data_list = request.json  # Теперь ожидаем список JSON объектов

    with open('demo/train_image_020_1.png', 'rb') as f:
        orig = base64.b64encode(f.read()).decode('utf-8')

    with open('demo/train_image_020.png', 'rb') as f:
        mask = base64.b64encode(f.read()).decode('utf-8')

    resp = {
        "orig": orig,
        "mask": mask,
    }
    return make_response(resp)

