import cv2
import pytesseract
import os


# Detecting Characters
def detect_characters(img):
    hImg,wImg,_ = img.shape
    boxes = pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(img,(x,hImg-y),(w,hImg-h),(0,0,255),1)
        cv2.putText(img,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
    return img


# Detecting Words
def detect_words(img):
    header = ''
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    hImg,wImg,_ = img.shape
    boxes = pytesseract.image_to_data(img)
    for x,b in enumerate(boxes.splitlines()):
        if x != 0:
            b = b.split()
            # Filter by confidence level
            if len(b) == 12 and int(b[10]) > 30:
                if header == '':
                    header = b[11]
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img,(x,y),(w+x,h+y),(0,0,255),1)
                cv2.putText(img,b[11],(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(50,50,255),1)
    cv2.waitKey(0)
    return img,header

