import cv2
##class ObjectDetection:
#   def __init__(self):
#       pass

#using cv2 dnn does not work due to some weird issues in the detect function
# i tried looking into it online and it seems like it could be related
# to openCV's compatibility with certain operating systems, 
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