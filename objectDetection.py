import cv2
##class ObjectDetection:
#   def __init__(self):
#       pass

def main():

    net = cv2.dnn.readNetFromONNX('yolov5/yolov5s.onnx')
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(640,640), scale=1/255)
    videoInput = cv2.VideoCapture(0)
    
    while True:
        ret, frame = videoInput.read()
       # blob = cv2.dnn.blobFromImage(frame, size=(640,640))
        if (ret == True):
            (classIds, scores, bboxes) = model.detect(frame)
            print("class ids", classIds)
            print("scores", scores)
            print("bboxes", bboxes)
    
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
    return 0

if __name__ == "__main__":
    main()