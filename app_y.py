from importlib import import_module
import os
import jsonpickle
from flask import Flask,Response , request , flash , url_for,jsonify
import logging
from logging.config import dictConfig
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask 
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
app = Flask(__name__)

@app.route('/') 
def hello_world():
    return "This API is running perfectly!"

@app.route('/classifier/run',methods=['POST'])
def classify():
    app.logger.debug('Running classifier')
    upload = request.files['data']
    IMG_SIZE=224
    model = load_model('/home/soumi/Downloads/best1.h5') 


    image_path="File2.jpg"
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)


    result=model.predict_classes(img)

    predicted_class = ("bridge", "child","tristep1","tristep2","tristep3")[result[0]]
    return(predicted_class)



if __name__ == '__main__':
    #load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
