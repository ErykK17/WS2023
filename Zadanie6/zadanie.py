import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


# Załaduj dane MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Eksploracja danych
print(f'Treningowe obrazy: {train_images.shape}')
print(f'Treningowe etykiety: {train_labels.shape}')
print(f'Testowe obrazy: {test_images.shape}')
print(f'Testowe etykiety: {test_labels.shape}')

# Wizualizacja kilku losowych obrazówep
plt.figure(figsize=(10, 4))
for i in range(24):
    plt.subplot(3, 8, i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.axis('off')
plt.show()

# Przygotowanie danych
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Budowa modelu CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Ewaluacja modelu
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Dokładność na zbiorze testowym: {test_acc}')
print(f'Utrata na zbiorze testowym: {test_loss}')

# Zapisz model
model.save('mnist_model.h5')

# Wizualizacja historii treningu
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

plt.show()

# Wizualizacja przykładowych błędnych prognoz
predictions = model.predict(test_images)
predicted_labels = tf.argmax(predictions, axis=1).numpy()
true_labels = tf.argmax(test_labels, axis=1).numpy()

misclassified_indices = (predicted_labels != true_labels)

plt.figure(figsize=(10, 10))
for i, idx in enumerate(range(24)):
    plt.subplot(3, 8, i+1)
    plt.imshow(test_images[misclassified_indices][idx].reshape(28, 28), cmap='gray')
    plt.title(f'True: {true_labels[misclassified_indices][idx]}\nPredicted: {predicted_labels[misclassified_indices][idx]}')
    plt.axis('off')
plt.subplots_adjust(hspace=0.5)
plt.show()


misclassified_count = sum(misclassified_indices)
total_samples = len(test_labels)
accuracy = 1 - (misclassified_count / total_samples)
print(f"Nietrafione: {misclassified_count}/{total_samples - misclassified_count} (Dokładność: {accuracy:.2%})")
