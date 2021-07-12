---
title:  "Air Mouse: Controlling Mouse With Gestures in Air"
date:   2021-07-12 10:29:17 +0545
categories:
  - Computer Vision
  - Image Processing
  - Project
tags:
  - Computer Vision
  - Image Processing
header:
  teaser: assets/air-mouse/thumbnail.png
---
# Air Mouse: Doing Mouse Operations Using Finger Gestures
Hey surfer, in this blog, I am going to write about how can we do basic mouse operations like move pointer, click, double click and right click using only finger gestures. 

This blog is the part of the series [#7DaysOfComputerVisionProjects](https://www.youtube.com/playlist?list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE). Links to the blogs and videos of each projects are:
1. **Real-time Background Changing**: [Video](https://www.youtube.com/watch?v=JZ9cIAlCh7c&list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE&index=2) | [Blog]({{site.url}}/2021/07/11/real-time-background-changing/)
2. **Air Mouse: Control Mouse with Gestures** [Video](https://www.youtube.com/playlist?list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE) | [Blog]({{site.url}}/2021/07/12/air-mouse-control-mouse-with-gestures/)
3. **Play Trex Game With Gesture** [Video](https://www.youtube.com/watch?v=70VjkDus22g&list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE&index=4) | [Blog]({{site.url}}/2021/07/13/playing-chrome-trex-game-with-gestures/)
4. **Auto Dino: Play Trex Game Automatically** [Video](https://www.youtube.com/watch?v=73lSzQcXRLg&list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE&index=5) | [Blog]({{site.url}}/2021/07/14/play-trex-with-image-processing/)
5. **Gesture Based Writing** [Video](https://www.youtube.com/watch?v=hjiaAv6zYVY&list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE&index=6) | [Blog]({{site.url}}/2021/07/15/gesture-based-visually-writing-system/)
6. **Game: Kill The Fly** [Video](https://www.youtube.com/playlist?list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE) | [Blog]({{site.url}}/2021/07/16/game-kill-a-fly/)
7. **Gesture Based Calculator** [Video](https://www.youtube.com/playlist?list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE) | [Blog]({{site.url}}/2021/07/17/gesture-based-calculator/)


## Introduction
As the project name Air Mouse, it is a Computer Mouse except working by the Gestures of fingers. We will be using 2 python libraries, mouse and Mediapipe. Mouse is a library to do mouse operations like click, drag, release and so on. We will be using [Hand Module of Mediapipe](https://google.github.io/mediapipe/solutions/hands.html) a OpenSource tool to extract the landmarks of hand and fingers. But it have multiple modules like selfie segmentation, pose estimation, face detection etc.

### Installation
It will be best idea to install these tools on virtual environment.
* `pip install mediapipe` for installing mediapipe. 
* `pip install mouse` for installing mouse package.


## Preliminary Tasks
### Import Dependencies


```python
import mediapipe as mp
import cv2
import mouse
import numpy as np
import tkinter as tk
```

### Get Screen Size
The use of `tkinter` is only to find screen size.


```python

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

ssize = (screen_height, screen_width)
ssize
```




    (768, 1366)



## Write Basic Functions
* We need to convert the landmark position from the frame world to our screen world thus the method `frame_pos2screen_pos` is written.
* We will be working with Euclidean Distance to make some sense about gestures. 


```python
def frame_pos2screen_pos(frame_size=(480, 640), screen_size=(768, 1366), frame_pos=None):
    x,y = screen_size[1]/frame_size[0], screen_size[0]/frame_size[1]    
    screen_pos = [frame_pos[0]*x, frame_pos[1]*y]
    return screen_pos

def euclidean(pt1, pt2):
    d = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
    return d
euclidean((4, 3), (0, 0))
```

    5.0

## Writing a Code

### Step By Step
It is necessary to view the landmark position before making a gesture assumptions. Please follow the below image.

>![img]({{site.url}}/assets/air-mouse/landmarks.png)
Source: [Official Hands Page](https://google.github.io/mediapipe/solutions/hands.html)

* Start by beginning a camera.
```python 
cam = cv2.VideoCapture(0)
```
* Define a frame size in our case 520 rows and 720 columns.
```python
fsize = (520, 720)
```
* For ROI i.e the Region of Interest to care, define a rectangle that resides on the some area inside the frame but make sure it will leave enough space on each side.
```python
left,top,right,bottom=(200, 100, 500, 400)
```
* Take modules `drawing_utilities` and `hands` from Mediapipe solutions's. As the name, `drawing_utils` will draw landmark here and the `hands` will let us work with detection models.
```python
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
```
* Define a variable to count the frame and make a constant to check the events on those frame count.
```python
check_every = 10
check_cnt = 0
```
* Prepare a variable to hold last event name.
```python
last_event = None
```
* Prepare a variable events, single click, double click, right click and drag.
```python
events = ["sclick", "dclick", "rclick", "drag"]
```
* Now prepare a Mediapipe Hand object by giving arguments like `max_num_hands`, `min_detection_confidence` and so on. As name suggests, `max_num_hands` is to search up to that number of hands and `min_detection_confidence` is the minimum confidence threshold value of detection and below which, detected hands are discarded.
```python
with mp_hands.Hands(
static_image_mode=True,
max_num_hands = 2,
min_detection_confidence=0.6) as hands:
```
* Read a Camera frame.
```python
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            continue
```
* Flip the frame to look like selfie camera.
```python
        frame = cv2.flip(frame, 1)
```
* Resize frame to our desired size.
```python
        frame = cv2.resize(frame, (fsize[1], fsize[0]))
```
* Make a rectangle to show ROI Area on frame.
```python
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
```
* Extract width and height of frame.
```python        
        h, w,_ = frame.shape
```
* Convert frame from BGR to RGB because `Hand` object expects image as a RGB format.
```python
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```
* Pass the RGB image ot `process` module of `Hand` object to get the result.
```python
        res = hands.process(rgb)
```
* Now for each hand, we will be extracting landmarks of fingers. Like index finger's tip, dip, middle and so on. There are overall 21 landmarks for each hand. After extracting, we need to convert it back to pixel coordinate world.
```python
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                
                index_dip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y, 
                    w, h)
                
                index_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y, 
                    w, h)
                
                index_pip = np.array(mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y, 
                    w, h))
                
                thumb_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y, 
                    w, h)
                
                middle_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, 
                    w, h)
```
* Now if the current count of frame is equal to the value we defined earlier, then check for events. 
```python
                if index_pip is not None:
                    if check_cnt==check_every:
```
* If the distance between index finger's tip and middle finger's tip is less than 60 then consider that event as *double click*. i.e touch index finger and middle finger for double click. **The value 60 will be relative to the frame size.** Else, if last event is also double click, then set last event to none.
```python
                        if thumb_tip is not None and index_tip is not None and middle_tip is not None:

                            #print(euclidean(index_tip, middle_tip))
                            if euclidean(index_tip, middle_tip)<60: # 60 should be relative to the height of frame
                                last_event = "dclick"
                            else:
                                if last_event=="dclick":
                                    last_event=None
```
* If the distance between index pip, and thumb tip is less than 60 then consider that event as *single click*. i.e move thumb near to the bottom of index finger for single left click. Else if last event is also single click, then set last event to none.
```python
                        if thumb_tip is not None and index_tip is not None:
                            if euclidean(thumb_tip, index_pip) < 60: # 60 should be relative to height/width of frame
                                last_event = "sclick"
                            else:
                                if last_event=="sclick":
                                    last_event=None
```
* If thumb tip and index finger tip distance is below 60 then consider that event as *left press*. i.e if thumb tip and index tip comes near do left button press. It will help us to do selection. Else if last event is also left press, then set last event to release. 
```python
                        if euclidean(thumb_tip, index_tip) < 60:
                                last_event="press"
                            else:
                                if last_event == "press":
                                    last_event = "release"       
```
* If thumb tip and middle finger tip distance is below 60 then consider that event as *right click*. i.e if thumb tip and middle finger tip comes near do right click. Else if last event is also right click, set last event to none.
```python
                    if thumb_tip is not None and index_tip is not None and middle_tip is not None:

                            if euclidean(thumb_tip, middle_tip)<60: # 60 should be relative to the height of frame
                                last_event = "rclick"
                            else:
                                if last_event=="rclick":
                                    last_event=None
```
* After checking all events, set frame count to 0.
```python
                check_cnt=0
```
* Convert our useful landmarks from entire frame world to screen world:
    * First clip the values to only ROI region.
    ```python
            index_pip[0] = np.clip(index_pip[0], left, right)
            index_pip[1] = np.clip(index_pip[1], top, bottom)
    ```
    * Convert clipped values to Frame World i.e treat top left of ROI as top left of frame and for entire coordinates.
    ```python
            # normalize the pip values
            index_pip[0] = (index_pip[0]-left)*fsize[0]/(right-left)
            index_pip[1] = (index_pip[1]-top)*fsize[1]/(bottom-top)
    ```
    * Convert frame world point of index pip to screen world point by doing simple unitary method. A method `frame_pos2screen_pos` will do it.
    ```python
            screen_pos = frame_pos2screen_pos(fsize, ssize, index_pip)
    ```
* Move the cursor to converted position i.e. index pip.
```python
            mouse.move(str(int(screen_pos[0])), str(int(screen_pos[1])))
```
* Finally, if current frame count has been reseted then apply the event. And increase the frame count.
```python
            if check_cnt==0:
                if last_event=="sclick":
                    mouse.click()
                elif last_event=="dclick":
                    mouse.double_click()
                elif last_event=="press":
                    mouse.press()
                elif last_event=="rclick":
                    mouse.right_click()
                else:
                    mouse.release()
                print(last_event)

            check_cnt+=1
```
* Draw each landmarks.
```python
mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
```
* Show the frame.
```python
cv2.imshow("Controller Window", frame)
```

### Complete Code


```python
cam = cv2.VideoCapture(0) 
fsize = (520, 720)

left,top,right,bottom=(200, 100, 500, 300)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


check_every = 10
check_cnt = 0

last_event = None
events = ["sclick", "dclick", "rclick", "drag"]

with mp_hands.Hands(
static_image_mode=True,
max_num_hands = 1,
min_detection_confidence=0.7) as hands:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (fsize[1], fsize[0]))
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        
        h, w,_ = frame.shape
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        res = hands.process(rgb)
        #cv2.imshow("roi", roi)
        rgb.flags.writeable = True
        
        
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                
                index_dip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y, 
                    w, h)
                
                index_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y, 
                    w, h)
                
                index_pip = np.array(mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y, 
                    w, h))
                
                thumb_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y, 
                    w, h)
                
                middle_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, 
                    w, h)
                
                if index_pip is not None:
                    if check_cnt==check_every:
                        if thumb_tip is not None and index_tip is not None and middle_tip is not None:

                            #print(euclidean(index_tip, middle_tip))
                            if euclidean(index_tip, middle_tip)<60: # 60 should be relative to the height of frame
                                last_event = "dclick"
                            else:
                                if last_event=="dclick":
                                    last_event=None


                        if thumb_tip is not None and index_tip is not None:
                            #print(euclidean(thumb_tip, index_pip))
                            if euclidean(thumb_tip, index_pip) < 60: # 60 should be relative to height/width of frame
                                last_event = "sclick"
                            else:
                                if last_event=="sclick":
                                    last_event=None

                            if euclidean(thumb_tip, index_tip) < 60:
                                last_event="press"
                            else:
                                if last_event == "press":
                                    last_event = "release"       
                        
                        if thumb_tip is not None and index_tip is not None and middle_tip is not None:

                            #print(euclidean(index_tip, middle_tip))
                            if euclidean(thumb_tip, middle_tip)<60: # 60 should be relative to the height of frame
                                last_event = "rclick"
                            else:
                                if last_event=="rclick":
                                    last_event=None



                        check_cnt=0

                    #print(index_pip)
                    index_pip[0] = np.clip(index_pip[0], left, right)
                    index_pip[1] = np.clip(index_pip[1], top, bottom)

                    # normalize the pip values
                    index_pip[0] = (index_pip[0]-left)*fsize[0]/(right-left)
                    index_pip[1] = (index_pip[1]-top)*fsize[1]/(bottom-top)


                    screen_pos = frame_pos2screen_pos(fsize, ssize, index_pip)

                    mouse.move(str(int(screen_pos[0])), str(int(screen_pos[1])))

                    if check_cnt==0:
                        if last_event=="sclick":
                            mouse.click()
                        elif last_event=="dclick":
                            mouse.double_click()
                        elif last_event=="press":
                            mouse.press()
                        elif last_event=="rclick":
                            mouse.right_click()
                        else:
                            mouse.release()
                        #print(last_event)
                                
                       

                    check_cnt+=1

#                     cv2.putText(frame, last_event, (10, 50),
#                                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Controller Window", frame)
        
        if cv2.waitKey(1)&0xFF == 27:
            break
cam.release()
cv2.destroyAllWindows()
```

### Better Version of Code


```python
cam = cv2.VideoCapture(0)

fsize = (520, 720)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

left, top, right, bottom = (200, 100, 500, 400)

events = ["sclick", "dclick", "rclick", "drag", "release"]

check_every = 15
check_cnt = 0
last_event = None

out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (fsize[1], fsize[0]))

with mp_hands.Hands(static_image_mode=True,
                   max_num_hands = 1,
                   min_detection_confidence=0.5) as hands:
    while cam.isOpened():
        ret, frame = cam.read()
        
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (fsize[1], fsize[0]))
        
        h, w, _ = frame.shape
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                index_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    w, h)
                
                index_dip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y, 
                    w, h)
                
                
                index_pip = np.array(mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y, 
                    w, h))
                
                thumb_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y, 
                    w, h)
                
                middle_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y, 
                    w, h)
            
                
                index_tipm = list(index_tip)
                index_tipm[0] = np.clip(index_tipm[0], left, right)
                index_tipm[1] = np.clip(index_tipm[1], top, bottom)
                
                index_tipm[0] = (index_tipm[0]-left) * fsize[0]/(right-left)
                index_tipm[1] = (index_tipm[1]-top) * fsize[1]/(bottom-top)
                
                
                if check_cnt == check_every:
                    if thumb_tip is not None and index_tip is not None and middle_tip is not None:
                        if euclidean(index_tip, middle_tip)<40:
                            last_event = "dclick"
                        else:
                            if last_event == "dclick":
                                last_event=None
                    if thumb_tip is not None and index_pip is not None:
                        if euclidean(thumb_tip, index_pip)<60:
                            last_event = "sclick"
                        else:
                            if last_event == "sclick":
                                last_event=None
                    if thumb_tip is not None and index_tip is not None:
                        if euclidean(thumb_tip, index_tip) < 60:
                            last_event = "press"
                        else:
                            if last_event == "press":
                                last_event="release"
                    if thumb_tip is not None and middle_tip is not None:
                        if euclidean(thumb_tip, middle_tip)<60:
                            last_event = "rclick"
                        else:
                            if last_event=="rclick":
                                last_event=None
                    check_cnt = 0

                
                if check_cnt>1:
                    last_event = None
                
                
                screen_pos = frame_pos2screen_pos(fsize, ssize, index_tipm)
                
                print(screen_pos)
                
                mouse.move(screen_pos[0], screen_pos[1])
                
                if check_cnt==0:
                    if last_event=="sclick":
                        mouse.click()
                    elif last_event=="rclick":
                        mouse.right_click()
                    elif last_event=="dclick":
                        mouse.double_click()
                    elif last_event=="press":
                        mouse.press()
                    else:
                        mouse.release()
                    print(last_event)
                    cv2.putText(frame, last_event, (20, 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                check_cnt += 1
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        cv2.imshow("Window", frame)
        out.write(frame)
        if cv2.waitKey(1)&0xFF == 27:
            break
cam.release()
out.release()
cv2.destroyAllWindows()
```

## Finally
The above code works but it is hard to get to the come point and do the desired operation within a while so it is still a bad system. I will be working on above system to try make it more efficient. If you found this blog helpful then please leave us a comment on our YouTube video and don't forget to subscribe us. The code is available on GitHub.
* [Code Link](https://github.com/data-coil/7-Days-Of-Computer-Vision-Projects/tree/main/2.%20Air%20Mouse)
* [YouTube Video Link](https://youtu.be/V-F94Pl8Bf0)
