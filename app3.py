import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Button, filedialog
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image

model = keras.models.load_model('base_model.h5')

classes = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
           'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

def predict_image(model, image_path, width):
    image = keras_image.load_img(image_path,
                                 color_mode='grayscale',
                                 target_size=(width, width))
    x = keras_image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    test_image = np.vstack([x])
    result = model.predict(test_image, batch_size=8)
    predicted_class = classes[np.argmax(result)]
    return predicted_class

def browse_image():
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")))
    image_path_label.config(text=filename)
    preview = keras_image.load_img(filename)
    plt.imshow(preview)
    plt.axis('off')
    plt.show()

def predict_selected_image():
    image_path = image_path_label.cget("text")
    if image_path:
        predicted_class = predict_image(model, image_path, 224)
        prediction_label.config(text="Prediction: " + predicted_class)
    else:
        prediction_label.config(text="Please select an image first.")

# GUI Setup
root = Tk()
root.title("Image Prediction")
root.geometry("400x200")

select_image_button = Button(root, text="Select Image", command=browse_image)
select_image_button.pack()

image_path_label = Label(root, text="")
image_path_label.pack()

predict_button = Button(root, text="Predict", command=predict_selected_image)
predict_button.pack()

prediction_label = Label(root, text="")
prediction_label.pack()

root.mainloop()
