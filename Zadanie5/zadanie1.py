import cv2

# Inicjalizacja detektora twarzy z wykorzystaniem klasyfikatora Haara
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Wczytaj obraz
image = cv2.imread('harold.jpg')

# Konwertuj obraz na odcienie szarości
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Wykryj twarze na obrazie
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

# Narysuj ramki wokół wykrytych twarzy
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Zapisz obraz z ramkami
cv2.imwrite('modified_image.png', image)

