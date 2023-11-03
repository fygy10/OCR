import easyocr
import cv2
import matplotlib.pyplot as plt

#read image
img_path = '/Users/tim/Local/Python/EasyOCR/Images/stop_sign.jpeg'
img = cv2.imread(img_path)

#setup read instance
reader = easyocr.Reader(['en'], gpu = False) #gpu False is default

#detect text
text_ = reader.readtext(img)
#print(text_)

threshold = .25
#draw bounding box
for t in text_:
    print(t)
    bbox, text, score = t   #print out order of information

    if score > threshold:   #threshold based on confidence index
        cv2.rectangle(img, bbox[0], bbox[2], (0,255,0), 5)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()





#general steps: read image, instance for text detector, detect text on image, draw box