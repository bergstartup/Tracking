import cv2
import numpy as np
import math
import imutils
import os
import face_recognition
import math

known_face_encodings=[]#To store encoding of persons

#Retriving known persons from db
faces = os.listdir('person')
for i in faces:
    face = face_recognition.load_image_file("person/" + i)
    known_face_encodings.append(face_recognition.face_encodings(face)[0])

ids=len(known_face_encodings)#For idetifying person

#Finding euclidian distance
def dist(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2 +(y2-y1)**2)

#Drawing box around detected objects
def drawbox(mid,img,center_x,center_y,h,w):
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)
    color=colors[mid]
    cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
    cv2.putText(img,str(mid),(x,y+10),font,3,color,2)

#Face identification and matching
def identify_face(image):
    #Locating face and cropping it
    mark= face_recognition.face_locations(image, number_of_times_to_upsample=1, model='cnn')
    if len(mark)==0:
        return -1
    face=image[mark[0][0]:mark[0][2],mark[0][-1]:mark[0][1]]
    try:
        encode=face_recognition.face_encodings(image, known_face_locations=mark, num_jitters=1)[0]
    except:
        pass
    #Checking with db
    global known_face_encodings
    try:
        if len(known_face_encodings)!=0:
            result=face_recognition.compare_faces(known_face_encodings,encode,tolerance=0.6)
            print(result)
            for i,j in enumerate(result):
                if j:
                    return i+1
    except:
        print("In comp")
        
    #Assigning new id for new face and saving it to db
    global ids
    ids+=1
    label="person/"+str(ids)+".jpg"
    cv2.imwrite(label,face)
    known_face_encodings.append(encode)
    return ids
    
#Video to frames
cap = cv2.VideoCapture('Resource_videos/class_track.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frameCount)


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(100, 3))

framedict={}#Ids with respective previous frame coordinates
threshold=5 #Max distance of a person from one frame to next frame
detected=[] #Detected ids in the current frame
appearence=10#A look back parameter
ar=[]#For collecting imgs to group that into video
count=0
height=0
width=0
try:
    #Processing through frames
    while True:
        #Reading images and identifying the end of frame
        ret,img=cap.read()
        count+=1
        #Down scaling of image
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
            
        #YOLO network
        #Normalising and forward pass
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    boxes.append([center_x,center_y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        #Non max supression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        detected=[] #Accquring detected ids in current frames
        #Tracking process
        for i in range(len(boxes)):
            #Checking for person
            if((i in indexes)and(class_ids[i]==0)):
                center_x, center_y, w, h = boxes[i] #Extrating coordinates for each identification
                mind=threshold #Providing threashold
                mid=-1 #Initial id
                #Mapping id for each person in current frame through euclidian distance principle
                for j in framedict.keys():
                    if(framedict[j][-1]>0)and(j not in detected):
                        d=dist(center_x,center_y,framedict[j][0],framedict[j][1])
                        if(d<=mind):
                            #print(d)
                            mind=d
                            mid=j
                            
                #If no id is matched with previous frame
                if mid==-1:
                    
                    #Cropping img
                    xmin=center_x-w//2
                    ymin=center_y-h//2
                    if xmin<0:
                        xmin=1
                    if ymin<0:
                        ymin=1
                    xmax=xmin+w
                    ymax=ymin+h
                    cropped=img[ymin:ymax,xmin:xmax]
                    mid=identify_face(cropped)

                print("mid : ",mid)
                if mid!=-1:
                    detected.append(mid) #Adding id to current frame detected list
                    framedict[mid]=[center_x,center_y,appearence] #Noting appearence of id
                    drawbox(mid,img,center_x,center_y,h,w) #Drawing box across

        for i in framedict.keys():
           if i not in detected:
               framedict[i][-1]=framedict[i][-1]-1 
        ar.append(img)#Appending processed frames
        height=img.shape[0]
        width=img.shape[1]
except Exception as e:
    print(e)

#Releasing the image buffer
cap.release()

#Writing into video
out = cv2.VideoWriter('result_video/output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15,(width,height))
for i in ar:
    out.write(i)
out.release()
