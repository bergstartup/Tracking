# Tracking
Person tracking using YOLO and face recognition

A python program to track people in videos. This can be well used for surveillance. Let's see briefly about the main blocks we use in the code.

<b>YOLO object detection</b><br>
YOLO is the acronym for YOU ONLY LOOK ONCE, it's a NN which can localize the objects in the given frame. It can identify several
objects but we use it only to localize persons, as of now let's consider only the output of YOLO. YOLO's output is vector which 
consist of center coordinate of the object[x,y], height[h] and width[w] and set of confidence scores[ci]. The class with greater
confidence is the predicted class of that object. For n objects identified in frame n such vectors are produced. Right now we want YOLO to identify only persons in the frame.

<b>Face recognition library</b><br>
We provide cropped photos of identified persons. Face recognition lib gives the coordinate of the face in that cropped image. The face is cropped and stored in a database or be compared with other faces.

<b>Putting the above libs together for perfect tracking</b><br>
Initially, we separate video to frames with the help of OpenCV. Then we iterate through each frame for tracking. On each iteration, YOLO outputs vectors in the above-mentioned format. We look only into the vectors that are of persons. On the first frame, we store faces of all the persons
and provide them a unique id. We store the current x<sup>t</sup>,y<sup>t</sup> coordinate [Center point] of the person [Stored in a dictionary {'id':'x,y'}].
On subsequent frames we again identify person using Yolo, get their coordinates [x<sup>t+1</sup>,y<sup>t+1</sup>]. This time we find the euclidean distance for current frame coordinates with previous frame coordinates. The current frame coordinate which is near to previous frame coordinate is considered to be the current position of that id. 

Since there might be a frame, where a person is entering and another person is leaving, if we apply the above algorithm there will be ambiguity. So we keep a threshold parameter. This threshold parameter says the max displacement a person can make in a single frame. If the min distance pair is not less than the threshold then, that will be considered as a new person.

<b>On conclusion</b><br>
This algorithm lags in a few cases If a person overlaps another If a person never shows his face since the beginning. Other than that it's perfectly fine.
Moreover, further development can be made in this, such as capturing multiple images of a person for more accurate result maintaining a log, using it
to a CCTV infrastructure, where the code runs collect logs of persons. 

<b>Dependencies</b><br>
Use python 3.x<br>
<h3>Libraries:<h3><br>
Opencv [pip3 install opencv-python]<br>
Face recognition [pip3 install face_recognition]
