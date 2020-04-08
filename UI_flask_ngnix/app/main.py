import os
import tensorflow as tf
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
#from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
#from tensorflow.keras.models import Sequential, load_model
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np
import json
import requests

app = Flask(__name__)

#global graph
#vgg16 = load_model('model/model_automate_workflow.h5')

#graph = tf.compat.v1.get_default_graph()
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (224, 224)
UPLOAD_FOLDER = 'uploads'
SERVER_URL = 'http://172.17.0.1:8501/v1/models/document:predict'



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  # After this point, all image pixels reside in [0,1).
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Crop the central region of the image with an area containing 87.5% of
  # the original image.
  image = tf.image.central_crop(image, central_fraction=0.875)
  # Resize the image to the height and width needed for VGG-16 model 
  # Here it's 224, but it can vary based on the architecture of the model used.
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(
      image, [224, 224], align_corners=False)
  image = tf.squeeze(image)
  return image

def predict(file):
    # img = Image.open(file)
    # img = img.resize(IMAGE_SIZE, Image.ANTIALIAS)
    # img = np.array(img)
    # img = np.expand_dims(img, axis=0)
    # with graph.as_default():
    #     probs = vgg16.predict(img)[0]
    # labels = ['Счет', 'Договор', 'Лицензия']

    # # #payload = { "instances": [{'input_image': img.tolist()}] }
    # # #response = requests.post(SERVER_URL, json=payload)
    # # #response_json= json.loads(response.text)
    # label_num = np.argmax(probs, axis=0)
    # output = labels[label_num]
    # output = 'счет'
    with open(file, 'rb') as f:
  # Read the image data from a JPEG image
      data = f.read()  
      im_data = preprocess_image(data)
      with tf.Session() as sess:
        sess.run(im_data)
        im_data_arr = im_data.eval()
      payload = {'signature_name': 'predict', "instances": [{'images': im_data_arr.tolist()}] }
      labels = ['Счет', 'Договор', 'Лицензия']
      r = requests.post(SERVER_URL, json=payload)
      r_json= json.loads(r.text)
      probs = r_json["predictions"][0]
      label_num = np.argmax(probs, axis=0)
      output = labels[label_num]
      return output

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=False, port=80)
