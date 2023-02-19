
#
#https://github.com/ultralytics/yolov5
#https://www.youtube.com/watch?v=Cof7CNjDppo&t=640s
import cv2
import torch
from time import time
import numpy as np

device = 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/yolov5s.onnx')
classes = model.names
def score_frame(frame):
    dim = [frame]
    results = model(dim)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:,:-1]
    #print("labels:", labels)
    #print("cord:", cord)
    return labels, cords

def class_to_label(label):
    return classes[int(label)]

def plot_box(labels, cords, frame):
    n = len(labels)
    x_shape, y_shape = frame.shape[0], frame.shape[1]
    
    for i in range(n):
        row = cords[i]
        if row[4] >= 0.2: #confidence
            print(row[0], row[1], row[2], row[3])
            x1,y1,x2,y2 = int(row[0]*x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0, 0))
    return frame

def main():
    videoInput = cv2.VideoCapture(0)
    print("names:", model.names)
    while True:
        ret, frame = videoInput.read()
        assert ret
        
        frame = cv2.resize(frame, (640,640))
        startTime = time()
        labels, cords = score_frame(frame)
        plot_box(labels, cords, frame)
        endTime = time()
        fps = 1/np.round(endTime - startTime, 2)
        #print("FPS:", fps)

        cv2.imshow('test', frame)
        cv2.waitKey(1)
    return 0

if __name__ == "__main__":
    
    main()