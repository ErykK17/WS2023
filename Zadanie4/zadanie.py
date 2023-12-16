import cv2
import numpy as np

# Krok 1: Wczytaj obraz i przekonwertuj go na obraz czarno-biały
vid = cv2.VideoCapture('myvideo.mp4')
ret, frame = vid.read()
cv2.imwrite('frame.jpg', frame)
image = cv2.imread('frame.jpg')
height, width = image.shape[:2]

# Definiuj obszar zainteresowania (ROI)
roi_bottom_left = (0, height)
roi_top_left = (0, height * 0.6)
roi_top_right = (width * 0.8, height * 0.6)
roi_bottom_right = (width, height)

roi_vertices = np.array([roi_bottom_left, roi_top_left, roi_top_right, roi_bottom_right], dtype=np.int32)

# Krok 2: Rozmycie przez filtr Gaussa
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Krok 3: Dobór konturów algorytmem Kenny'ego
edges = cv2.Canny(blurred_image, 50, 150)
cv2.imwrite('kenny.jpg', edges)

# Krok 4: Wybór określonego konturu na obrazie
mask = np.zeros_like(edges)
cv2.fillPoly(mask, [roi_vertices], 255)
masked_edges = cv2.bitwise_and(edges, mask)

contours, hierarchy = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
selected_contour = max(contours, key=cv2.contourArea)

# Krok 5: Rysuj kontury na obrazie
cv2.drawContours(image, [selected_contour], -1, (0, 0, 255), 2)

cv2.imwrite('modified_frame.jpg', image)
