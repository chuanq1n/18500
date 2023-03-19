
#used the following resources to setup yolov5s to process live video feed from camera
#https://github.com/ultralytics/yolov5
#https://www.youtube.com/watch?v=Cof7CNjDppo&t=640s
import cv2
import torch
from time import time
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchvision.models as tmod
#m = tmod.mobilnet_v2(pretrained=False)
#print("mobilnetv2", m)

#yolov5 export: 
# in yolov5 directory
# $ python export.py --weights yolov5n.pt --include onnx --opset 12
# onnx runs faster on cpu

confidenceThreshold = 0.6
device = 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/yolov5n.onnx')
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s') // using onnx model is much faster on cpu, im getting 1-2 fps on this
classes = model.names
#goTurnTracker = cv2.TrackerGOTURN_create()
deepSort = DeepSort(max_age=5,
                   n_init=2,
                   nms_max_overlap=1.0,
                   embedder='mobilenet',
                   half=True,
                   embedder_gpu=False)
#deepSort = DeepSort(max_age=5) #instantiate deepsort
#track track ids and bboxes with map
trackMap = {}
#track track ids and direction
dirMap = {} 
#used to label entrance line of each door in format id : [(x1,y1), (x2,y2)]
doors = {}
#count for each room
doorCounts = {}
doorid = 0

#video frame dimension globals
framex = 640
framey = 640
halfx = framex//2    
halfy = framey//2

def score_frame(frame):
    dim = [frame]
    results = model(dim)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:,:-1]
    return labels, cords

def class_to_label(label):
    return classes[int(label)]

def plot_box(labels, cords, frame):
    n = len(labels)
    x_shape, y_shape = frame.shape[0], frame.shape[1]
    detections = []
    for i in range(n):
        row = cords[i]
        if row[4] >= confidenceThreshold and int(labels[i]) == 0: #confidence is row[4]
            x1,y1,x2,y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            
            detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4].item(), 'person'))
            # these three lines are for printing out detection bounding boxes
            #bgr = (0, 255, 0)
            #cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
            #cv2.putText(frame, class_to_label(labels[i]), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr)
    return frame, detections

# finds the displacement of a person over last 10 frames
def updateTrackMap(id, center):
    offset = 3
    numFrames = 10 
    # center point coords
    if id in trackMap.keys():
        trackMap[id].append(center)
        if (len(trackMap[id]) > 10):
            #print("before modify", trackMap[id])
            trackMap[id].pop(0)
            #print("after modify", trackMap[id])
            avgpt1 = sum(trackMap[id][0:5]) / 5
            #print(avgpt1)
            avgpt2 = sum(trackMap[id][5:10]) / 5
            #print(avgpt2)
            slope = avgpt2 - avgpt1
            #print("dx, dy", slope[0], slope[1])
            
            if (slope[0] < (-1 * offset)):
                dirMap[id] = "left"
            elif (slope[0] > offset):
                dirMap[id] = "right"
            
    else:
        trackMap[id] = [center]
    return None

def selectDoors(event, x, y, flags, param):
    global doorid
    if (event == cv2.EVENT_LBUTTONDOWN):
        print("down coord", x, y)
        doors[doorid] = [(int(x),int(y))]
    elif (event == cv2.EVENT_LBUTTONUP):
        print("up coord", x, y)
        doors[doorid].append((int(x),int(y)))
        doorid += 1

def checkCrossDoor(track_id, center, coords):
    global halfx # this is the middle of the frame
    global doorid # highest doorid
    global doorCounts #map containing doorid and corresponding counts
    global doors #coordinates of doors
    '''
    case 1: center left of middle line
    case 2: center right of middle line
    '''
    return 0

def main():
    #videoInput = cv2.VideoCapture(0)
    path = '/Users/bli/Desktop/500/CV/test2.mp4'
    videoInput = cv2.VideoCapture(path)

    
    try:
        while True:
            ret, frame = videoInput.read()
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) #for video in incorrect dimension
            if ret == True:
                frame = cv2.resize(frame, (framex,framey))
                startTime = time()
                labels, cords = score_frame(frame)
                f, bbs = plot_box(labels, cords, frame)

                tracks = deepSort.update_tracks(bbs, frame=f)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()

                    bbox = ltrb
                    center = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
                    coords = np.array(bbox)
                    #print(coords)
                    #print("point", point)
                    # update map containing each track and the centerpoint in past 10 frames 
                    updateTrackMap(track_id, center)
                    checkCrossDoor(track_id, center, coords)
                    cv2.rectangle(f, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
                    cv2.putText(f, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)
                    #lines for testing dimensions
                    cv2.line(f, (halfx, 0),(halfx, framey), (213, 255, 52), 1)
                    #cv2.line(f, (0, 10), (10, 10), (255,0,0), 1)
                    cv2.circle(f, (int(center[0]), int(center[1])), 1, (255, 255, 0), -1)
                    if (track_id in dirMap.keys()):
                        cv2.putText(f, dirMap[track_id], (int(bbox[2]) - 10, int(bbox[1]) - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,255,0), 1)
                
                endTime = time()
                # fps calculation
                fps = 1/np.round(endTime - startTime, 2)
                cv2.putText(f, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                # draw doors
                for id in doors.keys():
                    coords = doors[id]
                    cv2.line(f, coords[0], coords[1], (213, 255, 52), 1)

                cv2.imshow('test', f)
                cv2.setMouseCallback('test', selectDoors)
                key = cv2.pollKey()
                if (key & 0xFF == ord('p')): #logic to pause video with keyboard input
                    key = cv2.waitKey(0)
                    
                    if (key & 0xFF == ord('c')):
                        key = cv2.waitKey(1)
                    elif (key & 0xFF == ord('q')):
                        break
                key = cv2.waitKey(1)
                #print("key", key)
            else:
                break
    except KeyboardInterrupt:
        print("ending task by interrupt")
        videoInput.release()
        cv2.destroyAllWindows()
        return 0
    #end of loop
    #print("trackMap", trackMap)
    print("ending task")
    videoInput.release()
    cv2.destroyAllWindows()
    return 0

   

if __name__ == "__main__":
    main()