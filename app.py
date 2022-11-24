import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.utils import img_to_array, load_img

model =tf.keras.models.load_model('model.h5', compile=False)

num_to_label = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

def processed_img(img_path):
    img=load_img(img_path,grayscale=True,target_size=(28, 28, 1))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = [num_to_label[round(x[0])] for x in answer]
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    res = y

    print(answer)
    print(res)
    return res

def run():
    st.title("Image Classification Using CNN")
    st.markdown(
        "Fashion MNIST is intended as a drop-in replacement for the classic MNIST dataset—often used as the "
        "Hello, World of machine learning programs for computer vision. The MNIST dataset contains images of handwritten digits (0, 1, 2, etc) in an identical format to the articles of clothing we'll use here.")

    image = Image.open('fashion.png')

    st.image(image, caption='Fashion Mist samples (by Zalando, MIT License)')

    st.subheader("This Image Classification Using CNN detects Fashion MNIST class")
    st.text("Kindly upload an image in jpg or png file format")
    img_file = st.file_uploader("Choose an Image:", type=["jpg", "png"])

    if img_file is not None:
        st.image(img_file,use_column_width=False)

        if st.button("Predict"):
            result = processed_img(img_file)
            st.success("Fashion MNIST class: "+result)

run()