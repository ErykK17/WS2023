import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Wczytaj zapisany model
loaded_model = load_model('./mnist_model.h5')

# Ścieżka do pliku z obrazem cyfry
image_path = './siedem.jpg'

# Wczytaj obraz i przekształć do odpowiedniego formatu
input_image = Image.open(image_path).convert('L')
input_image = input_image.resize((28, 28))
input_array = np.array(input_image) / 255.0
input_array = np.expand_dims(input_array, axis=0)

# Ewaluacja za pomocą wczytanego modelu
predictions_single = loaded_model.predict(input_array)
print(predictions_single)
predicted_label_single = np.argmax(predictions_single)

# Wydrukuj wyniki
print(f'Przewidziana etykieta dla obrazu: {predicted_label_single}')
