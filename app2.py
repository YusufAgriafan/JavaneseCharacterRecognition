import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image as keras_image

# Load the model
model = keras.models.load_model('base_model.h5')

# Define classes
classes = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
           'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

def predict_image(model, image, width):
    image = image.resize((width, width))
    x = keras_image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    test_image = np.vstack([x])
    result = model.predict(test_image, batch_size=8)
    predicted_class = classes[np.argmax(result)]
    return predicted_class

def predict_and_display(image_path, canvas, model, width):
    image = Image.open(image_path)
    image = image.convert('L')
    predicted_class = predict_image(model, image, width)
    messagebox.showinfo("Prediction", "Predicted class: " + predicted_class)


def open_file(canvas, model, width):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        predict_and_display(file_path, canvas, model, width)

def main():
    root = tk.Tk()
    root.title("Handwriting Recognition")

    canvas_width = 224
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_width, bg="white")
    canvas.pack()

    img = Image.new("RGB", (canvas_width, canvas_width), "white")
    draw = ImageDraw.Draw(img)

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=1)
        draw.ellipse([x1, y1, x2, y2], fill="black")

    canvas.bind("<B1-Motion>", paint)

    predict_button = tk.Button(root, text="Predict", command=lambda: save_and_predict(canvas, model, canvas_width))
    predict_button.pack()

    clear_button = tk.Button(root, text="Clear", command=lambda: clear_canvas(canvas, img, draw))
    clear_button.pack()

    def save_and_predict(canvas, model, width):
        file_path = "temp.png"
        img.save(file_path, "png")
        predict_and_display(file_path, canvas, model, width)

    def clear_canvas(canvas, img, draw):
        canvas.delete("all")
        draw.rectangle([0, 0, canvas_width, canvas_width], fill="white")

    menubar = tk.Menu(root)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Open", command=lambda: open_file(canvas, model, canvas_width))
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=filemenu)
    root.config(menu=menubar)

    root.mainloop()

if __name__ == "__main__":
    main()
