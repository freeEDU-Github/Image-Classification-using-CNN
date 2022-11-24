import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras import preprocessing
from keras.utils import img_to_array, load_img
import os
import tensorflow_hub as hub
import matplotlib.pyplot as plt

st.title("Image Classification Using CNN")
st.text("Kindly upload an image in jpg or png file format")

def main():
    file_uploaded = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    #model = tf.keras.models.load_model('model.h5', compile=False)
    #shape = ((28,28,1))
    #model = tf.keras.Sequential(hub[hub.KerasLayer(classifier_model, input_shape=shape)])
    img = load_img(image, grayscale=True, target_size=(28, 28, 1))
    img = img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis = 0)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    model = tf.keras.models.load_model('model.h5', compile=False)
    predictions = model.predict(img)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    return result

if __name__ == "__main__":
    main()

