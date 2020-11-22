# TransferL-Yoga

This repo demonstrates transfer learning on **VGG16** with a **yoga dataset** of images.

The model endpoints are then made accessible by creating **API** endpoints with **Flask**.

Then this application is hosted online using **docker** and **AWS EC2** instance.

## Agenda:

  - Train and test DL Model.
  - Create a Flask app
  - Dockerize
  - Host on AWS
    
## To train and test DL Model:

  1. Obtain the dataset. Upload it to drive.

  2. Train the model on collab. Check the accurracy. Check the variables as they vary from one notebook to another, eg, val_acc and val_accuracy.

  3. Test the model with images from test directory and see if it gives satisfiable result.

[Here!](https://github.com/Soumi7/TFNotebooks/blob/master/TransferLYoga.ipynb) is the link to my collab notebook.

Download the saved model **best1.h5**.

Save it in the same folder where you want to test inference logic. Here is my tf_inference.py code for inference logic:

```
import numpy as np
import cv2
import h5py
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
#install tensorflow==2.2.0-rc3. Check collab version.
```

Once this runs successfully, use same inference logic in the Flask app. Create a file called app_y.py to run the app on localhost:80.

```
from flask import Flask, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image img_to_array
from tensorflow.keras.preprocessing import image
import h5py

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
    #load_model()  # If you want to load model at the beginning once only
    app.run(host='0.0.0.0', port=80)
```

Add a helloworld path to check its running. Add a classifier path to run the model. It takes a single image as input and returns the classname as output, here the yoga pose.

Use this command to run the app on port 80.

```
sudo python3 app_y.py
```

Check if the app is running using a CURL request.

```
curl -X POST https://localhost:80/classifier/run -F “data=@tri2.jpg”
```

Pass several files and check.

Create a Dockerfile in the main folder.

```
FROM python:3.6-slim
COPY ./app_y.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./best1.h5 /deploy/
WORKDIR /deploy/
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app_y.py"]
```

Make a requirements.txt file add requirements including the version on you system. Check them before like inside collab notebook:

```python3
>>>import tensorflow as tf
>>>print(tf.__version__)
```

Here is my requirements.txt code:

```jsonpickle
tensorflow==2.2.0-rc3
numpy
flask
Pillow
```

To get a docker image:

```sudo docker build -t app_y-yoga .```

Now, the image is built and ready to be run. We can do this using the command:

```sudo docker run -p 80:80 app_y-yoga .```

Check if its running and giving right output:

```curl -X POST http://localhost:80/classifier/run -F “data=@tri2.jpg”```

Now, if you build a container and had some problems running it, like if hosted successfully but curl didnt work, delete the docker container. In case permission is denied, try this. The lsof command gives list of processes running.

```sudo systemctl daemon-reload
sudo systemctl restart docker
sudo lsof -i -P -n | grep 80
sudo kill <PID of process running on port 80> 
```

# Hosting docker container on AWS

Sign into the AWS console and search for EC2 in the search bar to navigate to EC2 dashboard.

Generate a Key-Pair from the option and change permission with:

```chmod 400 <keyname>.pem```

Click Launch Instance on the EC2 dashboard:

Choose the Amazon Machine Instance (AMI) from the list of options. An AMI determines the OS that the VM will be running.

We work with the t2.micro instance.

Navigate to Configure Security Group tab and add a new rule http which takes 80 automatically.

Click Review and Launch.

Launch icon will lead to a pop up seeking a confirmation on having a key-pair. Use the name of the key pair that was generated earlier and launch the VM.

Click view instances. Click on your instance and copy the public-dns name given below on the right.

Now ssh into the ec2 machine from local system terminal using the command with the field public-dns-name replaced with your ec2 instance name:

```
ssh -i SoumiBardhan.pem ec2-user@public-dns-name
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```

Incase you are uploading files from a folder, declare the key private there as well with:

```
chmod 400 <keyname>.pem
```

To copy files into the ec2 instance, run:

```
scp -i SoumiBardhan.pem file-to-copy ec2-user@public-dns:/home/ec2-user1
scp -i SoumiBardhan.pem app_y.py ec2-user@public-dns:/home/ec2-user1
scp -i SoumiBardhan.pem requirements.txt ec2-user@public-dns:/home/ec2-user1
scp -i SoumiBardhan.pem Dockerfile ec2-user@public-dns:/home/ec2-user1
scp -i SoumiBardhan.pem best1.h5 ec2-user@public-dns:/home/ec2-user1
```

Now ssh into the Amazon VM and run the same docker commands used before on system:

```sudo docker build -t app_y-yoga .
sudo docker run -p 80:80 app_y-yoga .
```

Now check with curl requests:

```
curl -X POST ec2-user@public-dns/classifier/run -F "data=@tri2.jpg"
```
So the API is live now. Anybody can access it with the above curl request.


