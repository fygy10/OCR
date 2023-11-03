import easyocr
import cv2
import numpy as np

#read image
img_path = '/Users/tim/Local/Python/EasyOCR/Images/stop_sign.jpeg'
img = cv2.imread(img_path)

#setup read instance
reader = easyocr.Reader(['en'], gpu = False) #gpu False is default

#detect text
results = reader.readtext(img)

threshold = .25

#draw bounding box and text overlay
for (bbox, text, score) in results:
    if score > threshold:
        #unpack bounding box
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        #draw rectangle
        img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 2)

        #put text
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, text, (top_left[0], top_left[1] - 10), font, 0.5, (255,0,0), 2)

#optionally, draw contour bounding boxes
#convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#find Canny edges
edged = cv2.Canny(gray, 30, 200)
#finding contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#draw contours
cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

#show the image with bounding boxes and overlays
cv2.imshow('Detected Text', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


