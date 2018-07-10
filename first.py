from flask import Flask
import urllib.request, json 
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf


global graph

graph = tf.get_default_graph()


app = Flask(__name__)

with urllib.request.urlopen("http://localhost/drupal/api/img") as url:
    data = json.loads(url.read().decode())
    path='http://localhost'+data[0]['field_upload_image']
    urllib.request.urlretrieve(path,'local.jpg')
   
@app.route('/')
def hello_world():
    with graph.as_default():
        from keras.models import load_model
        model = load_model('model.nw')
        
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        from skimage.io import imread
        from skimage.transform import resize
        import numpy as np
         
        #class_labels = {v: k for k, v in training_set.class_indices.items()}
        
        img = imread('local.jpg') 
        img = resize(img,(64,64)) 
        img = np.expand_dims(img,axis=0) 
        prediction = model.predict_classes(img)
        
        if prediction[0]==0:
            return 'cat'
        else:
            return 'dog'

       

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5050)        
        
