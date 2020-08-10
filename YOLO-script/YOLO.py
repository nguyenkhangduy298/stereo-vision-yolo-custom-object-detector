#coding=utf-8

import time
import cv2
import argparse
import numpy as np
import os
import datetime  


ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--image', required=True,
#                 help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    print("ClassID" + str(class_id))
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cap0 = cv2.VideoCapture(0)
# cap0.set(3,640)
# cap0.set(4,480)
cap1 = cv2.VideoCapture(2)
# cap1.set(3,480)
# cap1.set(4,480)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if ret0:
        cv2.imshow("Cam 0", frame0)
    if ret1:
        cv2.imshow("Cam 1", frame1)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame0_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame0)
        img_name1 = "opencv_frame1_{}.png".format(img_counter)
        cv2.imwrite(img_name1, frame1)
        print("Image written!")
        img_counter += 1

        #Start Object detection
        image = cv2.imread(img_name1)

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(args.weights, args.config)

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

        file_name = str(datetime.datetime.now()) + '.jpg';

        # cv2.waitKey(0)
        path = '/home/sonng9800/CapstoneProject/yolo_beginner-master/result/'

        flag_return = cv2.imwrite("result.png", image)
        if (flag_return):
            print("Saved Ok")
        else:
            print("Save Error")

cap0.release()
cap1.release()

cv2.destroyAllWindows()

cv2.destroyAllWindows()
