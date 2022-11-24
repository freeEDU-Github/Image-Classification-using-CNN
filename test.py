# Helper libraries
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


model =tf.keras.models.load_model('model.h5', compile=False)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

model =tf.keras.models.load_model('model.h5', compile=False)

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

for test_images, test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

def plot_image(img, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[img], true_labels[img], images[img]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(img, predictions_array, true_label):
    predictions_array, true_label = predictions_array[img], true_label[img]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


st.title("Image Classification Using CNN")
st.text("Kindly upload an image in jpg or png file format")
img_file = st.file_uploader("Choose an Image:", type=["jpg", "png"])

if img_file is not None:
    st.image(img_file,use_column_width=False)
    save_image_path = 'mnist_test_images'+img_file.name
    with open(save_image_path, "wb") as f:
        f.write(img_file.getbuffer())

    if st.button("Predict"):
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plot_image(img_file, test_labels, test_images)
        plt.subplot(1, 2, 2)
        plot_value_array(img_file, predictions, test_labels)
        #st.success("Fashion MNIST class: "+result)
