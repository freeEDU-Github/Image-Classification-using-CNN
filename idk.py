import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras import preprocessing
from keras.utils import img_to_array, load_img
from matplotlib import pyplot as plt

model =tf.keras.models.load_model('model.h5', compile=False)

def processed_img(img_path):
    img = load_img(img_path, grayscale=True, target_size=(28, 28, 1))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    model = tf.keras.models.load_model('model.h5', compile=False)
    predictions = model.predict(img)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    print(result)
    print(scores)
    return result

def run():
    st.title("Image Classification Using CNN")
    st.text("Kindly upload an image in jpg or png file format")
    img_file = st.file_uploader("Choose an Image:", type=["jpg", "png"])

    if img_file is not None:
        st.image(img_file,use_column_width=False)
        figure = plt.figure()
        #plt.imshow(img_file)
        plt.axis('off')
        result = processed_img(img_file)
        st.write(result)
        st.pyplot(figure)

run()