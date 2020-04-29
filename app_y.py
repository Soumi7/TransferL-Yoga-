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
    image = load_image(upload)
    
    model = load_model('best1.h5') 
    result=model.predict_classes(image)

    predicted_class = ("bridge", "child","tristep1","tristep2","tristep3")[result[0]]
    return(predicted_class)

def load_image(filename):    
    IMG_SIZE=224
    test_image = image.load_img(filename, target_size = (IMG_SIZE, IMG_SIZE))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    return test_image

if __name__ == '__main__':
    #load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)