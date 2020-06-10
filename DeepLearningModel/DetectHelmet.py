from imutils.video import VideoStream
import numpy as np
from imutils.video import FPS
import imutils
import time
import cv2
from keras.models import load_model

CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model.......................................................................")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

print('Loading helmet model.......................................................................')
Helmet_model = load_model('new_helmet_model.h5')
Helmet_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("[INFO] starting video stream................................................................")

cap = cv2.VideoCapture('vid1.mp4')
fps = FPS().start()
while True:

    try:

        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600, height=600)
        h = 600
	w = 600
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)

        detections = net.forward()  # getting the detections from the network

        persons = []
        motorbike = []
        for detection in detections.reshape(-1,7) :
            # extract the confidence associated with the prediction
            confidence = float(detection[2])
            
            # filter out weak detections by ensuring the confidence
            # is greater than minimum confidence
            if confidence > 0.5:
                
                # extract index of class label from the detections
                idx = int(detection[1])
                
                if idx == 15:
                    startX = int(detection[3] * w)
                    startY = int(detection[4] * h)
                    endX = int(detection[5] * w)
                    endY= int(detection[6] * h)
                    persons.append((startX, startY, endX, endY))

                if idx == 14:
                    startX = int(detection[3] * w)
                    startY = int(detection[4] * h)
                    endX = int(detection[5] * w)
                    endY= int(detection[6] * h)
                    
                    motorbike.append((startX, startY, endX, endY))


        xsdiff = 0
        xediff = 0
        ysdiff = 0
        yediff = 0
        p = ()

        for i in motorbike:
            mi = float("Inf")
            for j in range(len(persons)):
                xsdiff = abs(i[0] - persons[j][0])
                xediff = abs(i[2] - persons[j][2])
                ysdiff = abs(i[1] - persons[j][1])
                yediff = abs(i[3] - persons[j][3])

                if (xsdiff+xediff+ysdiff+yediff) < mi:
                    mi = xsdiff+xediff+ysdiff+yediff
                    p = persons[j]


            if len(p) != 0:

	            # display the prediction
	            label = "{}".format(CLASSES[14])
	            print("[INFO] {}".format(label))
	            cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), COLORS[14], 2)
	            y = i[1] - 15 if i[1] - 15 > 15 else i[1] + 15
	            cv2.putText(frame, label, (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[14], 2)
	            label = "{}".format(CLASSES[15])
	            print("[INFO] {}".format(label))

	            cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), COLORS[15], 2)
	            y = p[1] - 15 if p[1] - 15 > 15 else p[1] + 15

	            roi = frame[p[1]:p[1]+(p[3]-p[1])//4, p[0]:p[2]]
	            print(roi)
	            if len(roi) != 0:
	            	img_array = cv2.resize(roi, (50,50))
	            	gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
	            	img = np.array(gray_img).reshape(1, 50, 50, 1)
	            	img = img/255.0
	            	prediction = Helmet_model.predict_proba([img])
	            	cv2.rectangle(frame, (p[0], p[1]), (p[0]+(p[2]-p[0]), p[1]+(p[3]-p[1])//4), COLORS[0], 2)
	            	cv2.putText(frame, str(round(prediction[0][0],2)), (p[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

    except:
        pass

    cv2.imshow('Frame', frame)  # Displaying the frame
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): # if 'q' key is pressed, break from the loop
        break
    fps.update()

fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
cap.release()   # Closing the video stream 
