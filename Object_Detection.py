import cv2
import numpy as np
import time

img1 = cv2.imread('C:/Users/Akash Sagar/Downloads/yolo/YOLOv3-Custom-Object-Detection-main/YOLOv3-Custom-Object-Detection-main/test_images/2582.jpg', cv2.IMREAD_UNCHANGED)
#img1 = cv2.imread('C:/Users/Akash Sagar/Downloads/yolo/YOLOv3-Custom-Object-Detection-main/YOLOv3-Custom-Object-Detection-main/test_images/', cv2.IMREAD_UNCHANGED) 
#print('Original Dimensions : ', img1.shape)
 
scale_percent = 1000 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
#resize image
resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)

net = cv2.dnn.readNetFromDarknet("C:/Users/Akash Sagar/Downloads/yolo/YOLOv3-Custom-Object-Detection-main/YOLOv3-Custom-Object-Detection-main/yolov3_training.cfg", "C:/Users/Akash Sagar/Downloads/yolo/YOLOv3-Custom-Object-Detection-main/YOLOv3-Custom-Object-Detection-main/yolov3_training_last.weights")

classes = []
with open("C:/Users/Akash Sagar/Downloads/yolo/YOLOv3-Custom-Object-Detection-main/YOLOv3-Custom-Object-Detection-main/classes.txt", "r") as f:
    classes = f.read().splitlines()

#cap = cv2.VideoCapture('video4.mp4')
cap = 'C:/Users/Akash Sagar/Downloads/yolo/YOLOv3-Custom-Object-Detection-main/YOLOv3-Custom-Object-Detection-main/test_images/2582.jpg'
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    #_, img = cap.read()
    img = resized
    #img = cv2.imread("C:/Users/Akash Sagar/Downloads/yolo/YOLOv3-Custom-Object-Detection-main/YOLOv3-Custom-Object-Detection-main/test_images/2582.jpg")
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (0,0,0), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break

#cap.release()
cv2.destroyAllWindows()

