# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras.preprocessing import image as keras_image

# model = keras.models.load_model('base_model.h5')

# classes = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
#            'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

# def predict_image(model, image_path, width):
#     image = keras_image.load_img(image_path,
#                                  color_mode='grayscale',
#                                  target_size=(width, width))
#     x = keras_image.img_to_array(image)
#     x = np.expand_dims(x, axis=0)
#     test_image = np.vstack([x])
#     result = model.predict(test_image, batch_size=8)
#     predicted_class = classes[np.argmax(result)]
#     return predicted_class

# def predict_images_in_directory(model, directory_path, width):
#     test_images_paths = os.listdir(directory_path)
#     for path in test_images_paths:
#         image_path = os.path.join(directory_path, path)  # Menggabungkan directory path dengan file name
#         predicted_class = predict_image(model, image_path, width)
        
#         print("Image:", image_path)
#         print("Prediction:", predicted_class)
        
#         preview = keras_image.load_img(image_path)
#         plt.imshow(preview, cmap='gray')
#         plt.show()
#     print('Prediction Done')

# predict_images_in_directory(model, "datasetv2/prediction", 224)


import os
import numpy as np
import matplotlib.pyplot as plt
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

def predict_images_in_directory(model, directory_path, width):
    test_images_paths = os.listdir(directory_path)
    for path in test_images_paths:
        image_path = os.path.join(directory_path, path)
        predicted_class = predict_image(model, image_path, width)
        
        print("Image:", image_path)
        print("Prediction:", predicted_class)
        
        preview = keras_image.load_img(image_path)
        plt.imshow(preview, cmap='gray')
        plt.show()
    print('Prediction Done')

predict_images_in_directory(model, "datasetv2/prediction", 224)
