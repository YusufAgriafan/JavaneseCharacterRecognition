import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing import image as keras_image
from tkinter import *
from tkinter import filedialog

# Fungsi untuk memilih gambar
def browse_image():
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                          filetypes=(("Image Files", "*.png; *.jpg; *.jpeg"), ("All Files", "*.*")))
    entry.delete(0, END)
    entry.insert(END, filename)

# Fungsi untuk memprediksi gambar yang dipilih
def predict_image():
    image_path = entry.get()
    if image_path:
        image = keras_image.load_img(image_path, color_mode='grayscale', target_size=(224, 224))
        x = keras_image.img_to_array(image)
        x = np.expand_dims(x, axis=0)

        test_image = np.vstack([x])
        result = model.predict(test_image, batch_size=8)

        result_label.config(text=classes[np.argmax(result)])

        preview = plt_image.imread(image_path)
        plt.imshow(preview)
        plt.axis('off')
        plt.show()
    else:
        result_label.config(text="No image selected.")

# Load model
model_path = "E:/Pemrograman/Kecerdasan Buatan/Translate Aksara/base_model.h5"
model = load_model(model_path)

# Daftar kelas
classes = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
           'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

# GUI
root = Tk()
root.title("Image Prediction")
root.geometry("400x200")

# Label untuk memilih gambar
Label(root, text="Select Image:").pack(pady=5)

# Input untuk path gambar
entry = Entry(root, width=50)
entry.pack(pady=5)

# Tombol "Browse" untuk memilih gambar
Button(root, text="Browse", command=browse_image).pack(pady=5)

# Tombol "Predict" untuk memprediksi gambar
Button(root, text="Predict", command=predict_image).pack(pady=5)

# Label untuk menampilkan hasil prediksi
result_label = Label(root, text="")
result_label.pack(pady=5)

root.mainloop()
