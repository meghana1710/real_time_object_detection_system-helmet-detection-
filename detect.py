import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model
import pytesseract
try:
    from PIL import Image
except ImportError:
    import Image
from PIL import Image,ImageFilter
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
import easyocr  



net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")

model = load_model('helmet-nonhelmet_cnn.h5')
print('model loaded!!!')

cap = cv2.VideoCapture('video.mp4')
COLORS = [(0,255,0),(0,0,255)]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
 

fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter('output.avi', fourcc, 5,(888,500))
def ROI():
    

    img = cv2.imread('./cropped_image.jpg')
#convert my image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#perform adaptive threshold so that I can extract proper contours from the image
#need this to extract the name plate from the image. 
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    contours,h = cv2.findContours(thresh,1,2)

#once I have the contours list, i need to find the contours which form rectangles.
#the contours can be approximated to minimum polygons, polygons of size 4 are probably rectangles
    largest_rectangle = [0,0]
    for cnt in contours:

        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        if len(approx)==4: #polygons with 4 points is what I need.
            area = cv2.contourArea(cnt)
            if area > largest_rectangle[0]:
            #find the polygon which has the largest size.
                largest_rectangle = [cv2.contourArea(cnt), cnt, approx]

    x,y,w,h = cv2.boundingRect(largest_rectangle[1])
#crop the rectangle to get the number plate.
    roi=img[y:y+h,x:x+w]
#cv2.drawContours(img,[largest_rectangle[1]],0,(0,0,255),-1)
    plt.imshow(roi, cmap = 'gray')
    plt.show()
    return roi

def helmet_or_nohelmet(helmet_roi):
	try:
		helmet_roi = cv2.resize(helmet_roi, (224, 224))
		helmet_roi = np.array(helmet_roi,dtype='float32')
		helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
		helmet_roi = helmet_roi/255.0
		return int(model.predict(helmet_roi)[0][0])
	except:
			pass

ret = True

while ret:

    ret, img = cap.read()
    img = imutils.resize(img,height=500)
    # img = cv2.imread('test.png')
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    classIds = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            color = [int(c) for c in COLORS[classIds[i]]]
            # green --> bike
            # red --> number plate
            if classIds[i]==0: #bike
                helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
            else: #number plate
                x_h = x-60
                y_h = y-350
                w_h = w+100
                
                h_h = h+100
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                # h_r = img[max(0,(y-330)):max(0,(y-330 + h+100)) , max(0,(x-80)):max(0,(x-80 + w+130))]
                if y_h>0 and x_h>0:
                    h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                    c = helmet_or_nohelmet(h_r)
                    print(c)
                    if(c):
                        x1 = x 
                        y1 = y 
                        x2 = x+w
                        y2 = y+h
                        cropped_image = img[y1:y2, x1:x2]
                        #cropped_image = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
                        cv2.imwrite("cropped_image.jpg", cropped_image)
                        image_file = Image.open("./cropped_image.jpg")
                        image_file.save("cropped_image_1.jpg", quality=95)
                        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                        #roi = ROI()
                        #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                        #text = pytesseract.image_to_string(roi)
                        reader = easyocr.Reader(['en'])
                        output = reader.readtext(image_file)
                        print(output)
                        

                    cv2.putText(img,['helmet','no-helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                
                    cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)


    writer.write(img)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break

writer.release()
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
