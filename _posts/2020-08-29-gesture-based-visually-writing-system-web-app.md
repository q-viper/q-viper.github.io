---
title: 'Gesture Based Visually Writing System: Building a Web App Using Flask'
date: 2020-08-29T02:58:08+05:45
header:
  teaser: assets/wp-content/uploads/2020/08/Capture-1.png
categories:
  - Computer Vision
  - Programming
  - Projects
tags:
  - computer vision
  - flask
  - gesture recognition
  - python
  - web development
---
**Contents**
* TOC
{:toc}

## Introduction
If I have to write about part, then this is the 5th part or 4th version of <b>Gesture Based Visually Writing System</b>. If you have read my previous blogs on this same topic then you probably know that how much have I progressed from the early version. <b><i>Once my employer suggested me on interview that you must know how to make your code usable by non coder/public only then your skill can be useful.</i></b> I am applying same concept. What he meant was I must learn how to deploy my system to non coder. I am not a pro here but I am writing this deployment code while learning so I am expecting feedbacks, suggestions from readers. On 4th version of system, I wrote a code that was more organized by the concept of OOP. Now I want this system to be deployed. It all starts with what if and if you tried to work on what if, you always will learn new things. On this version, I did wrote some Flask Code for deployment of this project on Web App. I will try to make it simple here as much as possible. For the better understanding of this project requires <b>primary prerequisite to understand the concepts and algorithms used on this blog are to view my previous blogs</b>. Because I am explaining little this time. Follow the [Before Anything](#Before-Anything) section for more information.

### What Now?
* I will write a code using Flask to deploy previous system on Web App.

A simple work flow will be something like below.
![png]({{site.url}}/assets/contour-writing/web_flow.png)

I will write a class that will communicate with our existing system and then on every time returns the frames, canvas and detected text to routes.

### Before Anything
Before anything, I am requesting you to view my previous blogs and only then return to this blog. Because the blogs below are the earlier version of the codes and concepts. I am still using same concepts and also you will be amazed to see how much I have progressed. I am only going to describe what is happening here on very little. Hence, if you are new to this project then it is best idea to view them. Please view them serially.
* [Gesture Based Visually Writing System Using OpenCV and Python]({{site.url}}/2020/08/01/gesture-based-visually-writing-system-using-opencv-and-python/)
* [Gesture Based Visually Writing System: Adding Visual User Interface]({{site.url}}/2020/08/11/gesture-based-visually-writing-system-make-a-visual-user-interface/)
* [Gesture Based Visually Writing System: Adding Virtual Animationn, New Mode and New VUI]({{site.url}}/2020/08/14/gesture-based-visually-writing-system-adding-virtual-animation-new-mode-and-new-vui/)
* [Gesture Based Visually Writing System: Add Slider, More Colors and Optimized OOP code]({{site.url}}/2020/08/21/gesture-based-visually-writing-system-add-slider-more-colors-and-optimized-code/)

### How Come?
If you are here, then you either are fed up or curious with my posts on LinkedIn or Twitter about Gesture Based Writing System. But however you are here, I am here because of the support people gave me. 
Many people have suggesting me to share these blogs on Medium, reditt to get more feedbacks and support. But I have very limited amount of cellular data to use hence I publish it on my site then share on LinkedIn and Twitter.

## Credits
I want to give credits of this blogs to everyone on [LinkedIn](https://linkedin.com/in/ramkrishna-acharya-91a217183/) who reacted, shared and commented my previous blog and on [Twitter](https://twitter.com/QuassarianViper) also (most of retweets was from bots lol). I am very grateful that my [this LinkedIn post about previous version](https://www.linkedin.com/posts/ramkrishna-acharya-91a217183_opencv-computervision-python-activity-6699919193124548608-bJI-) got more than 4k reactions and nearly 60k views. I never thought that this will gain so much attention and here I am improving it again. Also my friend [Dip](#) keeps asking me about my next work because he always supports me.

## Motivation
I am highly motivated by the support people gave me on [LinkedIn](https://www.linkedin.com/posts/ramkrishna-acharya-91a217183_opencv-computervision-python-activity-6699919193124548608-bJI-). 

### Quote
<b>And why do we fall Bruce?</b>
> So that we can learn to pick ourselves up. - Thomas Wyane


## Contour Writing: Modifications
For running this system, I had to modify this system little bit. Only modification needed was on class `ContourWriting` and I am including it below.

### Initialization
```python
        ##########################
        ##########################
        self.final_window = cv2.imread("static/taking_avg.png")
        self.detected_text = "Nothing" 
```
* `final_window`: A image that will tell user that average is being taken.
* `detected_text`: What has been detected by passing on detector?
Final window looks like below:
![png]({{site.url}}/assets/contour-writing/taking_avg.png)

### Method: `__del__`
There are default constructor made along with object creation on Python and `__del__` is a destructor called upon when all the references of an object is deleted.
```python
    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()
```
When an object of our ContourWriting is deleted, we want to close our camera and destroy all opened windows of OpenCV (if any available).

### Method: `detector`
```python
    def detector(self):
        img = self.canvas.canvas.astype(np.uint8)
        op = pytesseract.image_to_string(img, lang="eng", nice="1")
        self.detected_text = op
```
Nothing strange happened here, only detected text is assigned to attribute.

### Method: `main`
I had modified this method by some lines only. You can see the [codes here too](#Current-Contour-Writing). To make our system run without any possible errors, I tried to wrap codes inside try/except block. Main focus must be given on the try/except block near bottom of the below code.
```python
    def main(self): 
        try:
            while True:
                (ret, frame) = self.cam.read()
                if ret:
                    self.key = cv2.waitKey(1) & 0xFF
                    frame = imutils.resize(frame, width=self.size[1])
                    frame = cv2.flip(frame, 1)
                    clone = frame.copy()
                    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
                    self.set_grays(gray)
                    self.size = frame.shape
                    
                    # if to take average and num frames on average taking is lesser than 
                    if self.num_frames<self.avg_frames and self.take_average==True:
                        self.running_average()
                        cv2.putText(clone, str(self.num_frames), (self.roi_boxes["mroi"][1], self.roi_boxes["mroi"][0]-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
                        self.num_frames+=1
                    else:
                        self.take_average=False
                        clone = self.find_contours(clone)
                        fmode = self.check_force_mode() 
                        vui = self.vui.update_vui(pointer=self.roi_pointer["vroi"], cpointer=self.roi_pointer["droi"])
                        if self.vui.hover is not None:
                            self.running_mode = self.vui.running_mode   
                        if  self.roi_counts["mroi"] is not None:
                            if self.roi_counts["vroi"] is not None:
                                if self.roi_counts["mroi"]-5 > self.roi_counts["vroi"] and fmode is not None:
                                    self.running_mode = fmode
                                else:
                                    self.running_mode = self.vui.running_mode       
                            else:
                                self.running_mode = fmode
                        self.perform_mode()
                        self.vui.running_mode=self.running_mode
                        canvas = self.canvas.update_window(mode=self.running_mode, 
                                                        pointer=self.roi_pointer["droi"]).astype(np.uint8)
                        self.final_window = self.get_window(canvas=canvas, vui=vui)
                        
                        self.roi_pointer["vroi"] = (-1, -1)
                        
                    clone = self.make_rectangles(clone)
                    self.clone = clone
                    if self.key==27:
                        self.cam.release()
                        cv2.destroyAllWindows()
                        
                    try:
                        ret, clone_jpeg = cv2.imencode('.jpeg', self.clone)
                        ret1, draw_jpeg =  cv2.imencode('.jpeg', self.final_window)
                        if ret:
                            self.clone = clone_jpeg.tobytes()
                            self.vui_frame=draw_jpeg.tobytes()
                            return clone_jpeg.tobytes(), draw_jpeg.tobytes(), self.detected_text
                    except:
                        pass
        except:
            try:
                ret, clone_jpeg = cv2.imencode('.jpeg', self.clone)
                ret1, draw_jpeg =  cv2.imencode('.jpeg', self.final_window)
                if ret:
                    self.clone = clone_jpeg.tobytes()
                    self.vui_frame=draw_jpeg.tobytes()
                    return clone_jpeg.tobytes(), draw_jpeg.tobytes(), self.detected_text
            except:
                pass
```
* Take clone image, where we have drawn rectangles, contours, pointer and texts. Encode it to JPEG. 
* Take final window of VUI/Canvas and Encode it to JPEG.
* Convert both encoded images to bytes.
* Return the both byte images and detected text.

Initially, our final_window will be set to our image saying taking average. And when averages has been taken then we can send it to our flask routes. I have also set some attributes too. Entire codes inside `main.py` is given below.

## Codes: main.py
Please try to ignore the comments and lengths of the code.
```python
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import imutils
import pytesseract
import time

class VUI:
    """
        A class for visual user interface. Recommended to use default parameters.
    """
    def __init__(self, icons_dir="icons/", window_size=(525, 700, 3), vui_part=20, max_count=15):
        """
            icons_dir: directory to use icons from.
            window_size: size of an vui window. Recommended to use equal of final window.
            vui_part: How much % of rows from top will be used to stack icons?
        
        """
        self.idir = icons_dir
        self.window = np.zeros(window_size).astype(np.uint8)
        self.size = window_size
        self.vui_part = int(vui_part/100 * self.size[0])
        self.dd_part = (self.vui_part, int(50/100 * self.size[0]))
        
        self.modes = [fname.split(".")[0] for fname in os.listdir(self.idir)]
        self.icon_size = (int(self.size[1]/len(self.modes)), self.vui_part) # c, r
        
        self.current_icons = []
        self.anim_scale = 0.5
        self.anim_color = [5, 15, 2]
        self.prev_mode = "move"
        self.current_mode = "move"
        self.running_mode = None 
        self.hover=None
        
        self.mode_count = 1
        self.max_count = max_count
        self.color_count = 1
        self.max_color = 5
        
        self.current_pointer = (100, 100)
        self.canvas_pointer = None
        
        self.draw_color = (0, 0, 255)
        self.previous_color = (0, 0, 255)
        self.current_color = (0, 0, 255)
        self.pointer_color = (100, 200, 200)
        self.point = (10, -3)
        self.colors=None
        
        self.icons = self.prepare_icons()
        self.get_window()
        
    def prepare_icons(self):
        """
            A method to prepare icons on initial frame.
            Method sets 4 new attributes.
            cols: List to store (y1, y2) of icon.
            icon_position: Dictionary to store (y1, y2) as key and corresponding image as value
            current_icons: A dictionary initialized with initial icons. Changed on every frame when cursor lies above it.
            mode_pos: Mode as key and its icon's (y1, y2) as value.
        """
        icons = []
        cols = np.linspace(0, self.size[1]-1, len(self.modes)+1).astype(np.int64)
        cols = [(cols[i], cols[i+1]) for i in range(len(cols)-1)]
        
        icon_pos = {}
        mode_pos = {}
        for i, image_name in enumerate(os.listdir(self.idir)):
            img = cv2.imread(self.idir+image_name)
            img = cv2.resize(img, (cols[i][1]-cols[i][0], self.vui_part))
            icon_pos[cols[i]] = img
            mode_pos[self.modes[i]] = cols[i]
        self.cols = cols   
        self.icon_position = icon_pos
        self.current_icons = icon_pos
        self.mode_pos = mode_pos
        
    def set_colors(self, col=None, new_colors=None):
        """
            A method to set colors when pointer lies above color icon.
            Initially used subset of {Red, Green, Blue}
            col:- column where current pointer lies.
            new_colors:- If to use other colors.
            
            Method returns list of available colors on dropdown menu. 
            Changes the draw color, pointer color upon condition meet.
        """
        # earlier pointer was clipped within the vui
        pointer = self.canvas_pointer
        pointer = (pointer[1], self.vui_part+ pointer[0])
        if new_colors is None:
            r = np.array([0, 0, 255])
            g = np.array([0, 255, 0])
            b = np.array([255, 0, 0])
            colors = [r, g, b]
            colors_new = [colors[i]+colors[i+1] for i in range(len(colors)-1)]
            colors.extend(colors_new)
            self.colors = colors
        else:
            self.colors = new_colors
        rows = np.linspace(self.dd_part[0], self.dd_part[1], len(self.colors)+1).astype(np.int64)
        rows = [(rows[i], rows[i+1]) for i in range(len(rows)-1)]
        self.color_pos = {}
        for row, color in zip(rows, colors):
            self.color_pos[row] = color
            if row[0]<=pointer[1]<row[1] and col[0]<=pointer[0]<col[1]:
                self.current_color = (color.tolist())
                if self.current_color == self.previous_color:
                    self.color_count+=1
                else:
                    self.previous_color=self.current_color
                    self.color_count = 1
                if self.color_count>=self.max_color:
                    self.draw_color=self.current_color
                self.pointer_color = (np.abs(np.array([200, 200, 100])-color).tolist())
            self.current_window[row[0]:row[1], col[0]:col[1]] = color
        
        return self.colors  
    def get_window(self):
        """
            A method to return a VUI window upon called. Sets pointer on VUI canvas.
        """
        self.current_window = np.zeros_like(self.window).astype(np.uint8)
        for col, img in self.current_icons.items():
            self.current_window[:self.vui_part, col[0]:col[1]] = img
        if self.running_mode == "color":
            self.set_colors(col=self.cols[self.modes.index("color")])
        if self.current_pointer is not None and self.current_pointer[0]>0:
            cv2.circle(self.current_window, (self.current_pointer[1], self.current_pointer[0]), self.point[0], self.pointer_color, self.point[1])
        
        return self.current_window
    def update_vui(self, pointer=(100, 100), cpointer=(10, 100)):
        """
            A method to update the entire VUI properties and state.
            pointer: Current pointer on VUI part.
            cpointer: Current pointer on Canvas.
            
            cpointer is useful when working with color mode.
        """
        self.current_pointer = pointer
        self.canvas_pointer = cpointer
        #print(pointer, canvas_pointer)
        current_icons = {}
        self.hover=None
        if pointer[0]<=self.vui_part:
            for col, mode in zip(self.cols, self.modes):
                icon = self.icon_position[col].copy()
                ishape = icon.shape
                
                #print(mode)
                if col[0]<pointer[1]<=col[1]:
                    # pointer is above this icon now animate it.
                    self.current_mode = mode
                    zeros_icon = np.zeros_like(icon).astype(np.uint8)
                    
                    f = self.anim_scale*self.mode_count
                    r = int(ishape[0] * f)
                    c = int(ishape[1] * f)
                    icon = cv2.resize(icon, (c, r))
                    if f > 1:
                        rd = int((r - ishape[0])/2)
                        cd = int((c - ishape[1])/2)
                        
                        zeros_icon[:, :] = icon[rd:ishape[0]+rd, cd:ishape[1]+cd] 
                    else:
                        rd = int((ishape[0] - r)/2)
                        cd = int((ishape[1] - c)/2)
                        rdd, cdd = 0, 0
                        if ishape[0]-rd-rd > r:
                            rdd=1
                        if ishape[1]-cd-cd > c:
                            cdd=1
                        #print(icon.shape, ishape, rd, abs(r-rd), cd, abs(c-cd))
                        zeros_icon[rd:ishape[0]-rd-rdd, cd:ishape[1]-cd-cdd] = icon[::] 
                            
                    current_icons[col] = zeros_icon.astype(np.uint8) + np.uint8(np.array(self.anim_color)*self.mode_count)
                    
                    
                    if self.prev_mode == self.current_mode:
                        self.mode_count += 1
                    else:
                        self.prev_mode = self.current_mode
                        self.mode_count = 1
                    if self.mode_count >= self.max_count:
                        self.running_mode = self.current_mode
                        self.mode_count = 1
                        self.hover = True
                        
                else:
                    current_icons[col] = icon
                
            self.current_icons = current_icons
        else:
            self.mode_count = 1
                    
        return self.get_window()
                                 
# vui = VUI()
# #show(vui.window)
# vui.update_vui()
# vui.update_vui()
# vui.update_vui(pointer=(200, 100))
# vui.update_vui(pointer=(200, 100))

class Canvas:
    def __init__(self, window_size=(525, 700, 3), draw_color=(100, 100, 100), 
                 pointer_color=(0, 0, 0), bg_color=(25, 25, 25), mode="move", 
                 point=(10, -3), vui=None, ssize=(300, 50, 3)):
        """
            A method to initialize canvas.
            window_size: size of a canvas window.
            draw_color: drawing color in RGB.
            pointer_color: pointer color in RGB.
            bg_color: background color in RGB.
            mode: running mode.
            point: tuple of (pointer radius, thickness)
            vui: VUI object.
            ssize: Slider's size.
        
        """
        self.size=window_size
        self.draw_color=draw_color
        self.pointer_color = pointer_color
        self.bg_color = bg_color
        self.window = np.zeros(self.size, dtype=np.uint8)
        self.canvas= self.window.copy()+bg_color
        self.mode = mode
        self.pointer = None
        self.point = point
        self.current_window = self.window+self.canvas
        self.vui = vui
        self.ssize = ssize
        self.sregion = ()
        
    def slider(self, size=(300, 30, 3), spoint=50, scolor=(100, 55, 100)):
        """
            A method to change the pointer size by moving a slider.
            size: size of slider region.
            spoint: slider point, generally row position of pointer.
            scolor: slider color
        """
        swidth=10
        #swidth=int(5/50*spoint)
        #swidth = np.clip(swidth, 5, spoint)
        swindow=np.zeros(self.size).astype(np.uint8)
        swindow[:self.ssize[0], 0:self.ssize[1]] += np.uint8([255, 255, 255])  
        r1 = np.clip(spoint-swidth, swidth, self.ssize[0]-swidth)
        r2 = np.clip(spoint+swidth, swidth, self.ssize[0]-swidth)
        spoint = int(10/50 * spoint)
        #print(r1, r2, spoint)
        
        
        swindow[r1:r2, :self.ssize[1]] = scolor
        self.point=(spoint, self.point[1])
        #cv2.imshow("slider", swindow.astype(np.uint8))
        return swindow.astype(np.uint8)   
    def clear(self):
        self.window = np.zeros(self.size, dtype=np.uint8)
        self.canvas= self.window.copy()+self.bg_color
    def update_window(self, mode, pointer=(400, 100)):
        """
            mode: running mode
            pointer: where is pointer now?
        """
        self.mode = mode
        self.vui.mode=mode
        self.pointer = pointer
        self.draw_color=self.vui.draw_color
        self.pointer_color = self.vui.pointer_color
        #self.pointer = (np.clip(self.vui.vui_part, pointer[0], self.size[0]), pointer[1])
        #print("c", self.draw_color)
        swindow = np.zeros(self.size).astype(np.uint8)
        #print(pointer)
        if 0<pointer[0]<self.ssize[0] and 0<pointer[1]<self.ssize[1]:
            swindow=self.slider(spoint=pointer[0])
            self.mode = "move"
            #self.pointer = (pointer[0], pointer[1]+self.ssize[1])
            #self.pointer_color = self.bg_color
        if self.mode == "draw":
            cv2.circle(self.canvas, (self.pointer[1], self.pointer[0]), self.point[0], self.draw_color, self.point[1])
            self.current_window = self.window+self.canvas+swindow
            cv2.circle(self.current_window, (self.pointer[1], self.pointer[0]), self.point[0], self.pointer_color, self.point[1])
            
        elif self.mode == "erase":
            cv2.circle(self.canvas, (self.pointer[1], self.pointer[0]), self.point[0], self.bg_color, self.point[1])
            self.current_window = self.window+self.canvas+swindow
            cv2.circle(self.current_window, (self.pointer[1], self.pointer[0]), self.point[0], self.pointer_color, self.point[1])
            
        else:
            self.current_window = self.window+self.canvas+swindow
            cv2.circle(self.current_window, (self.pointer[1], self.pointer[0]), self.point[0], self.pointer_color, self.point[1])
            
        #show(self.canvas)
        #show(self.current_window)
        return self.current_window
    
class ContourWriting:
    """
        A class to bind all other classes uses.
    """
    def __init__(self, count_mode=10, avg_frames=100, 
                 rois={"droi":[200, 400, 430, 681],
                       "mroi":[80, 10, 150, 225], 
                       "vroi":[100, 400, 200, 681]}, 
                 icons_dir="icons/", aweight=0.5):
        """
            rois: types of ROIS(draw, move, vui)
        
        """
        self.aweight = aweight
        self.avg_frames=avg_frames
        self.roi_boxes = rois
        self.roi_averages = {key:None for key in rois.keys()}
        self.roi_grays = {key:None for key in rois.keys()}
        self.roi_masks = {key:None for key in rois.keys()}
        self.roi_pointer = {key:None for key in rois.keys()}
        self.roi_counts = {key:None for key in rois.keys()}
        self.size = (525, 700)
        self.set_pointer()
        self.vui = VUI()
        self.canvas_shape = (self.size[0]-self.vui.vui_part, self.size[1], 3) 
    
        self.canvas = Canvas(window_size=self.canvas_shape, vui=self.vui, bg_color=[255, 255, 255])
        self.running_mode = self.vui.running_mode
        
        self.force_modes=None
        self.fcount_mode=count_mode
        self.fcurrent_count=0
        self.fprev_mode = "move"
        self.check_force_mode()
        self.cam = cv2.VideoCapture(0)
        self.clone = None
        self.vui_frame = None
        self.num_frames = 0
        self.take_average=True
        self.final_window = cv2.imread("static/taking_avg.png")
        self.detected_text = "Nothing"

    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()
        #pass
    def set_pointer(self):
        for rname, pointer in self.roi_pointer.items():
            top, right, bottom, left = self.roi_boxes[rname]
            self.roi_pointer[rname] = (int((left+right)/2), int((top+bottom)/2))
    def running_average(self):
        for rname, roi in self.roi_averages.items():
            gimg = self.roi_grays[rname]
            if roi is None:
                roi = gimg.copy().astype("float")
            else:
                cv2.accumulateWeighted(gimg, roi, self.aweight)
            self.roi_averages[rname] = roi
    def set_grays(self, gray_frame):
        for rname, box in self.roi_boxes.items():
            top, right, bottom, left = box
            gray_roi = gray_frame[top:bottom, right:left]
            #gray_roi = cv2.bilateralFilter(gray_roi, 9, 15, 15)
            gray_roi=cv2.GaussianBlur(gray_roi, (7, 7), 0)
            self.roi_grays[rname] = gray_roi
            
    def make_rectangles(self, clone):
        cv2.putText(clone, f"Curr. Mode: {self.running_mode}", (self.roi_boxes["vroi"][1], self.roi_boxes["vroi"][0]-20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)           
        # make rectangle for everything, add text on middle of it
        for rname, box in self.roi_boxes.items():
            top, right, bottom, left = box
            mid = int((top+bottom)/2), int((left+right)/2) 
            if rname == "droi":
                cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(clone, rname, (mid[1], mid[0]),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            if rname == "mroi":
                
                cv2.rectangle(clone, (left, top), (int((left + right)/3), bottom), (0, 255, 0), 2)
                cv2.rectangle(clone, (int((left + right)/3), top), (2*int((left + right)/3), bottom), (0, 255, 0), 2)
                cv2.rectangle(clone, (2*int((left + right)/3), top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(clone, str("Mv"), (int((right)/1), int((top+bottom)/2)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(clone, str("Dr"), (int((left + right)/3), int((top+bottom)/2)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(clone, str("Er"), (2*int((left + right)/3), int((top+bottom)/2)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if rname == "vroi":
                gb_indices = int((left-right)/len(self.vui.modes))
                gb_indices = np.arange(right, left, gb_indices)
                gb_indices[-1] = gb_indices[-1]+1
                for i in range(len(gb_indices)-1):
                    _gleft = gb_indices[i]
                    _gright = gb_indices[i+1]
                    cv2.rectangle(clone, (_gleft, top), (_gright, bottom), (255, 0, 255), 3)
                    cv2.putText(clone, self.vui.modes[i][:2], (_gleft+2, int((top+bottom)/2)),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
        return clone
    
    def find_contours(self, clone, threshold=10):
        self.roi_counts = {key:None for key in self.roi_counts.keys()}
        for rname, ravg in self.roi_averages.items():
            # abs diff betn img and bg
            top, right, bottom, left = self.roi_boxes[rname]
            diff = cv2.absdiff(ravg.astype("uint8"), self.roi_grays[rname])    
            _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            (cnts, _) = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            m = (-1, -1)
            if len(cnts)!=0:
                max_cnt = max(cnts, key=cv2.contourArea)
                cv2.drawContours(clone, [max_cnt+(right,top)], -1, (0, 0, 255))   
                sshape = max_cnt.shape
                new_segmented = max_cnt.reshape(sshape[0], sshape[-1])
                m = new_segmented.min(axis=0)
                cv2.circle(clone, (right+m[0], top+m[1]), 15, self.vui.pointer_color, -3)
                self.roi_counts[rname] = len(max_cnt)
                # translate m of this roi to window shape
                #if len(max_cnt)>10:    

                if rname!="mroi":
                    pshape = self.size
                    if rname=="vroi":
                        pshape = (self.vui.vui_part, self.vui.size[1])
                    if rname=="droi":
                        # make it self.canvas.shape
                        #pshape = (self.canvas_shape[0], self.canvas_shape[1]-self.canvas.ssize[1]) 
                        pshape=self.canvas_shape
                    h = bottom - top
                    l = left - right


                    m = (int((m[0]/l)*pshape[1]), int((m[1]/h)*pshape[0]))    
                else:
                    m = (right+m[0], top+m[1])
                        #print(m)
                #else:
                        #m=(-1,-1)
                 #       pass
                #if rname=="droi":
                 #   m = (m[0]+self.canvas.ssize[1], m[1])
                self.roi_pointer[rname]=(m[1], m[0])
                
        return clone    
    
    def get_window(self, canvas, vui):
        final_window = vui.copy()
        canvas_cpy = canvas.copy()
        vshape = vui.shape
        cshape = canvas.shape
        #print(self.running_mode)
        if self.running_mode == "color":
            # get part where color lies and make those part of canvas_bg black
            cp = self.vui.mode_pos[self.running_mode]
            canvas[:self.vui.dd_part[1]-self.vui.vui_part, cp[0]:cp[1]] = vui[self.vui.vui_part:self.vui.dd_part[1], cp[0]:cp[1]]
            
            #show(canvas)
        #else:#
        final_window[self.vui.vui_part:, :] = canvas
        cp = self.roi_pointer["droi"]
        cp = (cp[1], cp[0]+self.vui.vui_part)
        point = self.canvas.point
        cv2.circle(final_window,  cp, point[0], self.canvas.pointer_color, point[1])
        return final_window
    
    def check_force_mode(self):
        top, right, bottom, left = self.roi_boxes["mroi"]
        if self.force_modes is None:
            x=np.linspace(right, left, 4).astype(np.int64)
            x=[(x[i],x[i+1]) for i in range(len(x)-1)]
            force_modes = ["move", "draw", "erase"]
            force_modes = {x[i]:force_modes[i] for i in range(len(x))}
            #print(force_modes)
            self.force_modes = force_modes
        elif self.roi_pointer["mroi"][0]>0:
            mpointer = self.roi_pointer["mroi"]
            
            for col, mode in self.force_modes.items():
                
                if col[0]<=mpointer[1]<col[1]:
                    #print(col, mpointer)
                    if self.fprev_mode==mode:
                        self.fcurrent_count+=1
                    else:
                        self.fcurrent_count=0
                        self.fprev_mode=mode
                    if self.fcurrent_count>=self.fcount_mode:
                        #print("f ", mode)
                        #self.fcurrent_count=0
                        
                        return mode
    def detector(self):
        img = self.canvas.canvas.astype(np.uint8)
        op = pytesseract.image_to_string(img, lang="eng", nice="1")
        self.detected_text = op
        #print("Detected: ", op)
    def perform_mode(self):
        if self.running_mode=="clear":
            self.canvas.clear()
            self.running_mode="move"
        if self.running_mode=="restart":
            self.take_average =True
            self.num_frames=0
            self.running_mode="move"
            self.canvas.clear()
        if self.running_mode=="save":
            #cv2.imshow("canvas", self.canvas.canvas.astype(np.uint8))
            cv2.imwrite(f"canvas {time.time()}.png", self.canvas.canvas.astype(np.uint8))
            #cv2.destroyWindow("canvas")
            self.running_mode="move"
        if self.running_mode=="exit":
            self.key=27
        if self.running_mode=="detect":
            self.running_mode="move"
            self.detector()
    def main(self): 
        try:
            while True:
                (ret, frame) = self.cam.read()
                if ret:
                    self.key = cv2.waitKey(1) & 0xFF
                    frame = imutils.resize(frame, width=self.size[1])
                    frame = cv2.flip(frame, 1)
                    clone = frame.copy()
                    gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
                    self.set_grays(gray)
                    self.size = frame.shape
                    #print(self.num_frames)
                    
                    # if to take average and num frames on average taking is lesser than 
                    if self.num_frames<self.avg_frames and self.take_average==True:
                        self.running_average()
                        cv2.putText(clone, str(self.num_frames), (self.roi_boxes["mroi"][1], self.roi_boxes["mroi"][0]-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
                        self.num_frames+=1
                    else:
                        self.take_average=False
                        clone = self.find_contours(clone)
                        fmode = self.check_force_mode() 
                        #vui = self.vui.get_window()
                        vui = self.vui.update_vui(pointer=self.roi_pointer["vroi"], cpointer=self.roi_pointer["droi"])
                        #self.roi_pointer["vroi"] = (-1, -1) # uncomment this and works on both but color pallete is not disabled
                        #self.running_mode = self.vui.running_mode # uncomment this when nothing works       
                        if self.vui.hover is not None:
                            self.running_mode = self.vui.running_mode   
                        if  self.roi_counts["mroi"] is not None:
                            if self.roi_counts["vroi"] is not None:
                                if self.roi_counts["mroi"]-5 > self.roi_counts["vroi"] and fmode is not None:
                                    self.running_mode = fmode
                                else:
                                    self.running_mode = self.vui.running_mode       
                            else:
                                self.running_mode = fmode
                        self.perform_mode()
                        self.vui.running_mode=self.running_mode
                        canvas = self.canvas.update_window(mode=self.running_mode, 
                                                        pointer=self.roi_pointer["droi"]).astype(np.uint8)
                        self.final_window = self.get_window(canvas=canvas, vui=vui)
                        #cv2.imshow("CW", final_window)
                        
                        self.roi_pointer["vroi"] = (-1, -1)
                        #print(vui.shape)
                        #print(canvas.shape)
                        #cv2.imshow("Canvas", canvas)
                        #cv2.imshow("VUI", vui)
                    clone = self.make_rectangles(clone)
                    self.clone = clone
                    #cv2.imshow("Feed", clone)
                    if self.key==27:
                        self.cam.release()
                        cv2.destroyAllWindows()
                        #self.__del__()
                        #break
                    try:
                        #self.detected_text="hurray"
                        ret, clone_jpeg = cv2.imencode('.jpeg', self.clone)
                        ret1, draw_jpeg =  cv2.imencode('.jpeg', self.final_window)
                        #print(ret, ret1)
                        if ret:
                            self.clone = clone_jpeg.tobytes()
                            self.vui_frame=draw_jpeg.tobytes()
                            return clone_jpeg.tobytes(), draw_jpeg.tobytes(), self.detected_text
                    except:
                        pass
        except:
            try:
                ret, clone_jpeg = cv2.imencode('.jpeg', self.clone)
                ret1, draw_jpeg =  cv2.imencode('.jpeg', self.final_window)
                #print(ret, ret1)
                if ret:
                    self.clone = clone_jpeg.tobytes()
                    self.vui_frame=draw_jpeg.tobytes()
                    return clone_jpeg.tobytes(), draw_jpeg.tobytes(), self.detected_text
            except:
                pass
        

        

#gw = ContourWriting(avg_frames=150, aweight=0.5)
#gw.main()

```

## Flask App
I don't know how to begin about Flask because I have done only few projects on Flask. If you have not installed Flask, then please install it using you package installer conda/pip. I will try to explain little bit of Flask codes but if you are new to flask, try to learn some basics about it too. I am learning Flask from e-books.

### Project Structure
* icons
    * icon images
* static
    * css
        * css files
    * error/average taking images
* templates
    * index.html
    * detection.html
* app.py
* main.py

## Import Dependencies
```python
from flask import Flask, render_template, Response
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import *
import time


err_img=cv2.imread("static\error_frame.png")
_, err_img_byte = cv2.imencode(".jpg", err_img)
err_img_byte = err_img_byte.tobytes()
draw=None
```
* Flask is used to create an app object.
* render_template is to render an html file on distinct route.
* Response is to send a Response object to http request.
* From our `main.py` file, we import everything but importing only class `ContourWriting` will be enough.

The image `error_frame.png` looks like below.
![png]({{site.url}}/assets/contour-writing/error_frame.png)

While showing everyframes taken from camera to the webpage, there were sometimes unexpected errors like can not grab a frame. On those situations, it wil be a great idea to show an error image instead of showing error. Also this image is converted to bytes because we are sending this content as content of Response later.

## Class: GestureWeb
As stated earlier on the flow diagram, this class will communicate with our ContourWriting class and takes current frames, VUI frame, and detected text. We initialize the class with only few things.
* Set `detected_text` to `Nothing` initially.
* Create an object of `ContourWriting`.

```python
class GestureWeb:
    def __init__(self, port=5005, debug=True):
        self.detected_text = "Nothing"
        self.camera= ContourWriting()
        
    def frame_gen(self, camera, kind="frame"):        
        while True:    
            frame = camera.main()
            if frame is None:
                frame = (None, None)
            if frame[0] is None:
                frame = err_img_byte
            elif frame[1] is None:
                draw = err_img_byte
            else:
                frame, draw, gw.detected_text = frame
                
            frame = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+bytearray(frame)+b'\r\n')
            draw = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+bytearray(draw)+b'\r\n')
            if kind =="frame":
                yield frame
            if kind =="draw":
                yield draw

```
### Method: `frame_gen`
This method is actually a generator method. This method calls the `main` method of the ContourWriting object and returns it to the calling route on every frame. Since we have to show every frames from our live feed to our app, we obviously need a thread to do this task without any execution delay. To return each frame to a calling function, we have to use the `yield` instead of the `return`. 
* Call the `main` method of `ContourWriting` object.
* If it is `None` then we will pass error images to frame and VUI part of our web app.
* Else, we divide the returned value from `main` method into 3 parts, `frame`,`draw`, `detected_text`.
* We then attach detected_text to attribute.
* We convert the JPEG encoded image into bytearray and attach it with our Content-Type.
* Then we yield the corresponding value (frame or VUI).

## Index File: index.html
I want this file to show running feed and the VUI side by side on same page. Hence I must find a way to stack then horizontally. I got help from stackoverflow's comment sections. I am not going to explain the CSS/HTML here because I am not good at them. 

{% raw %}
```HTML
<head>
    <title>
        Tindex
    </title>
    <style>
        .container{
            display: flex;
            flex-direction: row;
            flex-grow: 1;
        }
        .photos{
            display: flex;
            justify-content: center;
            flex-direction: column;
            flex-grow: 1;
        }
        .image{
            display: block;
            width: 100%;
        }
        .word{
            display: block;
            width: 100%;
            text-align: center;
        }
    </style>
    <body>
    <h1>Contour Based Writing System</h1>

    <div class="container">
        <div class="photos">
            <img class="image" src = "{{url_for('video_feed')}}">
            <span class = "word">Frames</span>
        </div>
        <div class="photos">
            <img class="image" src = "{{url_for('get_canvas')}}">
            <span class = "word">Canvas</span>
        </div>
    </div>
    <hr>
    <div>
        <iframe src="detection.html" style="float: inline-end;"></iframe>
    </div>
    
</body>
</head>

```
{% endraw %}
To view video feed from camera, we will use image instead of video. It may seem like the video by use but truth is, combination of photos is also a video. So I used image as frame, we won't know that it is image but we feel like a realtime video. And the speed of feed on web app is same as of camera. The `video_feed` is a route that returns the frame as Response object. I am not going to explain much because I am also learning. We have distinct route for distinct image, for frame or live feed we have `video_feed` and for canvas we have `get_canvas` and the Jinja has made it possible. So basically, we will have live feed on left side and our drawing canvas on right side.

To show the detected text, we have `detection.html` file. We are showing it on the bottom of our page and we are using it on same page by `iframe` tag.

## Detection File: detection.html
On this file we only show the `detected_text` attribute of the object of class `GestureWeb`. 

{% raw %}
```HTML
<!DOCTYPE html>
<html>
    <head>
        <meta http-equiv="refresh" content="5">
        <title>
            detection
        </title>
    </head>
    <body>
        <p><h3>Detected:</h3>{{detected_text}}</p>
    </body>
</html>
```
{% endraw %}

I want this page to refresh in every 5 seconds so that it will return the detected text under 5 seconds.

## Prepare Flask Decorators/routes
I hope someone comes up with a better way of doing following operations. I found these ways of working by searching on internet for whole day because this is my first Flask project from scratch. 

### Before First Request
```python
app = Flask(__name__)

@app.before_first_request
def before_first_request_func():
    global gw 
    gw = GestureWeb()
```

* Create a flask app using Flask class first.
* Use decorator `before_first_request` to do some tasks before the first request made to our app.
    * We made our `GestureWeb`'s object globally accessible.
    * By making our `GestureWeb`'s object on early stage of app creation also makes our camera ready.

### Main Route
What do we want to show when user enters the main url? 
```python
app.route('/')
def index():
    return render_template('index.html')
```
We want to render the `index.html` when user enters the main url. And we want to show everything on this page.


### Route: `video_feed`
As I stated earlier on the `index.html` there were 2 sources defined inside a `img` tag of each image. One of them is `video_feed`. 

```python
@app.route('/video_feed')
def video_feed():
    fresp = gw.frame_gen(gw.camera, kind="frame")
    return Response(fresp, mimetype='multipart/x-mixed-replace; boundary=frame')
```
On above code, `gw.frame_gen(gw.camera, kind="frame")`, `gw` is object of GestureWeb. The method `frame_gen` is generator. We are passing the object of `ContourWriting` and the kind of thing we are expecting on return. Finally we return the response type via object of Response.

### Route: `get_canvas`
As I stated earlier on the `index.html` there were 2 sources defined inside a `img` tag of each image. One of them is `get_canvas`. 

```python
@app.route('/get_canvas')
def get_canvas():
    fresp = gw.frame_gen(gw.camera, "draw")    
    return Response(fresp, mimetype='multipart/x-mixed-replace; boundary=frame')
```
On above code, `gw.frame_gen(gw.camera, kind="draw")`, `gw` is object of GestureWeb. The method `frame_gen` is generator. We are passing the object of `ContourWriting` and the kind of thing we are expecting on return.

### Route: `detection.html`
Now to make our `iframe` work on `index.html` file, we also need to make a route for our detection file. While rendering this file, we will send the current detected text too. Please look again onto `detection.html` file for more clearance.
```python
@app.route('/detection.html', methods=['GET', 'POST'])
def detect():
    return render_template("detection.html", detected_text=gw.detected_text)
    
```


### Teardown Request
This decorator is helpful when we close our tab or error occurs.
```python
@app.teardown_request
def teardown_request_func(error=None):
    global gw
    gw = GestureWeb()   
    return "ok"
```

### Run it
The standard way of running flask app is like below.
```python
if __name__=='__main__':    
    app.run(debug=True)
```

## Codes: app.py
```python
from flask import Flask, render_template, Response
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import *
import time


err_img=cv2.imread("static\error_frame.png")
_, err_img_byte = cv2.imencode(".jpg", err_img)
err_img_byte = err_img_byte.tobytes()
draw=None



class GestureWeb:
    def __init__(self, port=5005, debug=True):
        self.detected_text = "Nothing"
        self.camera= ContourWriting()
        
    def frame_gen(self, camera, kind="frame"):        
        while True:    
            frame = camera.main()
            if frame is None:
                frame = (None, None)
            if frame[0] is None:
                frame = err_img_byte
            elif frame[1] is None:
                draw = err_img_byte
            else:
                frame, draw, gw.detected_text = frame
            frame = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+bytearray(frame)+b'\r\n')
            draw = (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'+bytearray(draw)+b'\r\n')
            if kind =="frame":
                yield frame
            if kind =="draw":
                yield draw
            if kind=="text":
                yield detected_text

    
    
  
app = Flask(__name__)

@app.before_first_request
def before_first_request_func():
    global gw 
    gw = GestureWeb()
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_canvas')
def get_canvas():
    fresp = gw.frame_gen(gw.camera, "draw")    
    return Response(fresp, mimetype='multipart/x-mixed-replace; boundary=frame')

    
@app.route('/video_feed')
def video_feed():
    fresp = gw.frame_gen(gw.camera, kind="frame")
    return Response(fresp, mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/detection.html', methods=['GET', 'POST'])
def detect():
    return render_template("detection.html", detected_text=gw.detected_text)
    
@app.teardown_request
def teardown_request_func(error=None):
    global gw
    gw = GestureWeb()   
    return "ok"



if __name__=='__main__':    
    app.run(debug=True)
    
    
```


## Finally
The result of entire code should look like below. From any terminal run the `app.py` file. I am using VS Code and I have r
<figure>
<video src = "{{site.url}}/assets/contour-writing/web app.mp4" width="100%" controls autoplay loop> </video>
<figcaption style = "text-align:left; font-style:italic">Gesture writing on Web App</figcaption>
</figure> 

This was just a simple deployment on web app. There are numerous system where user can write on their canvas by clicking mouse button and predicting what has been written. Also some people were telling me that we can do this by simple addons also. But I am not trying to prove my methods are best one and should be widely used. I applied what I thought and I implemented ideas onto codes. Now I deployed implementation code to web app. There are still number of problems that I have not found and I hope someone will find it soon. I am eagerly waiting to reply every queries here not only on the LinkedIn. 

I am going to point out features and shortcomings of this system now. 

### Codes
Codes to current version of the system is available on link below and if it is not, then hit the comment or leave me message (LinkedIn or Twitter or mail me).
* [Contour Based Writing: Web APP](https://github.com/q-viper/Contour-Based-Writing/contour-based-writing-web)

### Features
* Deployed on web app.
* Live feed and drawing region are placed side by side.
* Detected text is checked per 5 seconds.

### Shortcomings
Shortcomings are most helpful to find new feature on next version. Well here are plenty of them.
* Sometimes frames are not read properly and OpenCV's warning keeps popping on the console.
* The system is usable by contour of anything. Hence some gesture confirming model must be used.
* Also on above video, there can be seen pointer moving rapidly on the VUI regions. It is not always acceptable by users. Hence it must be eliminated soon.

### Ideas
* What if we can create a model that can classify gesture and we can define a certain gesture for certain mode?
* What if we made a drawing model that can auto complete our drawing.

## What Next?
I will try to solve shortcomings on next time. But I am interested to make this system run on mobile phones too. As per now, I am thinking of taking frames from device camera and process it. Then use some API call to get those frame. I might use Unity. I am highly excited to try using LSTMs and other state of the art Deep Learning Algorithms to make this system more awesome but I don't have internet access (other than cellular data) to do broad research.

### Why not read more?
* [Gesture Based Visually Writing System Using OpenCV and Python]({{site.url}}/2020/08/01/gesture-based-visually-writing-system-using-opencv-and-python/)
* [Gesture Based Visually Writing System: Adding Visual User Interface]({{site.url}}/2020/08/11/gesture-based-visually-writing-system-make-a-visual-user-interface/)
* [Gesture Based Visually Writing System: Adding Virtual Animationn, New Mode and New VUI]({{site.url}}/2020/08/14/gesture-based-visually-writing-system-adding-virtual-animation-new-mode-and-new-vui/)
* [Gesture Based Visually Writing System: Add Slider, More Colors and Optimized OOP code]({{site.url}}/2020/08/21/gesture-based-visually-writing-system-add-slider-more-colors-and-optimized-code/)
* [Linear Regression from Scratch]({{site.url}}/2020/08/07/writing-a-linear-regression-class-from-scratch-using-python/)
* [Writing Popular ML Optimizers from Scratch]({{site.url}}/2020/06/05/writing-popular-machine-learning-optimizers-from-scratch-on-python/)
* [Feed Forward Neural Network from Scratch]({{site.url}}/2020/05/30/writing-a-deep-neural-network-from-scratch-on-python/)
* [Convolutional Neural Networks from Scratch]({{site.url}}/2020/06/05/convolutional-neural-networks-from-scratch-on-python/)
* [Writing a Simple Image Processing Class from Scratch]({{site.url}}/2020/05/30/image-processing-class-from-scratch-on-python/)
* [Deploying a RASA Chatbot on Android using Unity3d]({{site.url}}/2020/08/04/deploying-a-simple-rasa-chatbot-on-unity3d-project-to-make-a-chatbot-for-android-devices/)
* [Naive Bayes for text classifications: Scratch to Framework]({{site.url}}/2020/03/04/text-classification-using-naive-bayes-scratch-to-the-framework/)
* [Simple OCR for Devanagari Handwritten Text]({{site.url}}/2020/02/25/building-ocr-for-devanagari-handwritten-character/)

