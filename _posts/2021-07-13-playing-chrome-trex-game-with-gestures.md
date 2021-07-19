---
title:  "Playing Chrome Trex Game with Gestures"
date:   2021-07-13 10:29:17 +0545
categories:
  - Computer Vision
  - Image Processing
  - Project
tags:
  - Computer Vision
  - Image Processing
header:
  teaser: assets/trex-game/thumbnail-gest.png
---

# Play Trex Game on Chrome By Gesture Using OpenCV and Mediapipe
Hello there surfer! 
Since few days, I am thinking about some cool projects that can be done within some hours using Mediapipe and OpenCV in Python.
In this blog, I am writing about how can we play the popular trex game by only moving our fingers in front of the camera. Many of us have played this game but none of us were interested to play. 🤣🤦‍♂️🤦‍♂️ Well in this blog, we are going to play with with full intention.

This blog is the part of the series [#7DaysOfComputerVisionProjects](https://www.youtube.com/playlist?list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE). Links to the blogs and videos of each projects are:
1. **Real-time Background Changing**: [Video](https://www.youtube.com/watch?v=JZ9cIAlCh7c&list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE&index=2) | [Blog]({{site.url}}/2021/07/11/real-time-background-changing/)
2. **Air Mouse: Control Mouse with Gestures** [Video](https://www.youtube.com/playlist?list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE) | [Blog]({{site.url}}/2021/07/12/air-mouse-control-mouse-with-gestures/)
3. **Play Trex Game With Gesture** [Video](https://www.youtube.com/watch?v=70VjkDus22g&list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE&index=4) | [Blog]({{site.url}}/2021/07/13/playing-chrome-trex-game-with-gestures/)
4. **Auto Dino: Play Trex Game Automatically** [Video](https://www.youtube.com/watch?v=73lSzQcXRLg&list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE&index=5) | [Blog]({{site.url}}/2021/07/14/play-trex-with-image-processing/)
5. **Gesture Based Writing** [Video](https://www.youtube.com/watch?v=hjiaAv6zYVY&list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE&index=6) | [Blog]({{site.url}}/2021/07/15/gesture-based-visually-writing-system/)
6. **Game: Kill The Fly** [Video](https://www.youtube.com/playlist?list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE) | [Blog]({{site.url}}/2021/07/16/game-kill-a-fly/)
7. **Gesture Based Calculator** [Video](https://www.youtube.com/playlist?list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE) | [Blog]({{site.url}}/2021/07/17/gesture-based-calculator/)


## Prerequisties
* Mediapipe: Install it using `pip install mediapipe`.
* OpenCV: It will be installed by default while installing `mediapipe`. 
* Keyboard: Not the physical one because we are simulating key events using gestures. `pip install keyboard`.

Once installed make sure you can use them. Just import them and see if any error pops up.


```python
import mediapipe as mp
import cv2
import numpy as np
import keyboard
```

Using `keyboard` package is pretty easy just like `mouse` package. For test, we are going to simulate `down` and then `!echo hey`. I am using Jupyter Notebook hence I have to use `!` to use windows commands.


```python
# lets simulate down key and hello world
keyboard.press_and_release("!,e,c,h,o,space,h,e,y")
```


```python
# !ECHO HEY
```

We can even use keys like `control`.


```python
# lets simulate down key and hello world
keyboard.press_and_release("h,e,l,l,o,ctrl+a")
```

HELLO

The focus is not on the `Keyboard` package but to play a dino game. We will do something like gesture recognition based on the distance between certain landmarks. So lets define a method to find Euclidean distance.



```python
def euclidean(pt1, pt2):
    d = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
    return d
euclidean((4, 3), (0, 0))
```




    5.0



## Writing a Code

### Step By Step
It is necessary to view the landmark position before making a gesture assumptions. Please follow the below image.

![img]({{site.url}}/assets/air-mouse/landmarks.png)
Source: [Official Hands Page](https://google.github.io/mediapipe/solutions/hands.html)

* Start by beginning a camera.
```python 
cam = cv2.VideoCapture(0)
```
* Define a frame size in our case 520 rows and 720 columns.
```python
fsize = (520, 720)
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
* Now for each hand, we will be extracting landmarks of fingers. Like index finger's tip, dip, middle and so on. There are overall 20 landmarks for each hand. After extracting, we need to convert it back to pixel coordinate world.
```python
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                
                index_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y, 
                    w, h)
                
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
                if index_tip is not None:
                    if check_cnt==check_every:
```
* If the distance between index finger's tip and middle finger's tip is less than 60 then consider that *space* is pressed. i.e touch index finger and middle finger for Jump. **The value 60 will be relative to the frame size.** Else, if last event is also Jump, then set last event to none.
```python
                        if index_tip is not None and middle_tip is not None:

                            if euclidean(index_tip, middle_tip)<40: 
                                last_event = "jump"
                            else:
                                if last_event=="jump":
                                    last_event=None
```
* If the distance between index pip, and thumb tip is less than 60 then consider that *down* key is pressed. i.e move thumb near to the bottom of index finger for duck. Else if last event is also duck, then set last event to none.
```python
                        if thumb_tip is not None and index_tip is not None:
                            if euclidean(thumb_tip, index_pip) < 60: # 60 should be relative to height/width of frame
                                last_event = "duck"
                            else:
                                if last_event=="duck":
                                    last_event=None
```
* After checking all events, set frame count to 0.
```python
                    check_cnt=0
```
* * Finally, if current frame count has been reseted then apply the event. And increase the frame count.
```python
            if check_cnt==0:
                if last_event=="jump":
                    keyboard.press_and_release("space")
                elif last_event=="duck":
                    keyboard.press("down")
                else:
                    keyboard.release("down")
                print(last_event)


            check_cnt+=1
```
* Draw each landmarks.
```python
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)```
* Show the frame.
```python
        cv2.imshow("Controller Window", frame)
```
* If Escape is pressed then close.
```python
        if cv2.waitKey(1)&0xFF == 27:
            break
```

## Final Code



```python
cam = cv2.VideoCapture(0)
fsize = (520, 720)

last_event = None
check_cnt = 0
check_every = 5

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands




with mp_hands.Hands(
static_image_mode=True,
max_num_hands = 1,
min_detection_confidence=0.6) as hands:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (fsize[1], fsize[0]))
        
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
                
                if index_tip is not None:
                    if check_cnt==check_every:
                        if index_tip is not None and middle_tip is not None:

                            
                            if euclidean(index_tip, middle_tip)<40: # 60 should be relative to the height of frame
                                last_event = "jump"
                            else:
                                if last_event=="jump":
                                    last_event=None
                        
                        if thumb_tip is not None and index_tip is not None:
                            print(euclidean(index_tip, middle_tip))
                            if euclidean(thumb_tip, index_tip) < 60:
                                last_event="duck"
                            else:
                                if last_event == "duck":
                                    last_event = None
                        check_cnt=0
                
                if check_cnt==0:
                    if last_event=="jump":
                        keyboard.press_and_release("space")
                    elif last_event=="duck":
                        keyboard.press("down")
                    else:
                        keyboard.release("down")
                    print(last_event)

                check_cnt+=1

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Controller Window", frame)
        
        if cv2.waitKey(1)&0xFF == 27:
            break
cam.release()
cv2.destroyAllWindows()
                        
```

## Finally
This is the end of our blog and I hope you learned something valuable from here. Please let me know if you found any problems or errors. There is a video version of this blog and you can watch this on YouTube too.

* [GitHub](https://github.com/data-coil/7-Days-Of-Computer-Vision-Projects/tree/main/3.%204.%20Trex%20Game)
* [YouTube](https://www.youtube.com/playlist?list=PLUqDn7JaCwaTbqegRNfRZmBZSxiTtL8bE)

