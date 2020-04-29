import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
IMG_SIZE=224
model = load_model('/home/soumi/Downloads/best1.h5') 


image_path="File2.jpg"
img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)


result=model.predict_classes(img)

predicted_class = ("bridge", "child","tristep1","tristep2","tristep3")[result[0]]
print(predicted_class)


#install tensorflow==2.2.0-rc3




