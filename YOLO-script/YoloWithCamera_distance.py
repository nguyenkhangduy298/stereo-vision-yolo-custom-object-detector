#coding=utf-8

import time
import cv2
import argparse
import numpy as np
import datetime
import math


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--camera', required=True,
                help='path to config the camera input')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

classes = args.classes
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    # print("ClassID" + str(class_id))
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Begin the camera
cap = cv2.VideoCapture(int(args.camera))
cap.set(3,640)
cap.set(4,480)
scale = 0.00392
net = cv2.dnn.readNet(args.weights, args.config)

pixel_width = 1

while(True):
    try:
        # Capture frame-by-frame
        ret, image = cap.read()
        # Our operations on the frame come here
        # Display the resulting frame

        Width = image.shape[1]
        Height = image.shape[0]
        # cv2.imshow('frame',image)
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # Thực hiện xác định bằng HOG và SVM
        start = time.time()

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    pixel_width = w
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        # distance = (3 * 640)/ (2 * pixel_width * math.tan(29))
        distance = (3 * 670) / pixel_width
        # distance = 441.6/(0.20072727272727273 * pixel_width) # (focal *  realWidth * 640)/(pixelWidth * distance)


        print("Pixel width: ",pixel_width)
        print("Distance to object: ",distance)

        cv2.putText(image, "%.2fcm" % distance,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 3)
        cv2.imshow("Result", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except ZeroDivisionError:
        print("You can't divide by zero!")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

