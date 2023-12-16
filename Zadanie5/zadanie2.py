import cv2

# Inicjalizacja klasyfikatorów HOG
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Wczytaj obraz
image = cv2.imread('napoli.jpeg')

# Konwertuj obraz na odcienie szarości
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Wykryj twarze na obrazie
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

# Dla każdej wykrytej twarzy
for (x, y, w, h) in faces:
    # Narysuj prostokąt wokół twarzy
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Uzyskaj obszar zainteresowania (ROI) dla twarzy
    roi_gray = gray_image[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    # Wykryj oczy na twarzy
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Wykryj uśmiech na twarzy
    smiles = smile_cascade.detectMultiScale(roi_gray)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 2)

# Wykryj twarze z profilu na obrazie
profile_faces = profile_face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

# Dla każdej wykrytej twarzy z profilu
for (x, y, w, h) in profile_faces:
    # Narysuj prostokąt wokół twarzy z profilu
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Liczba wykrytych osób
num_people = len(faces) + len(profile_faces)
print(f"Liczba wykrytych osób: {num_people}")


# Zapisz obraz z wykrytymi cechami
cv2.imwrite('image_with_features.jpg', image)


