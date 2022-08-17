---
title:  "Building Image Size Reducer In Python"
date:   2022-08-14 02:29:17 +0545
categories:
    - Image Processing
    - Web App
tags:
    - Image Size Reducer
header:
  teaser: assets/image_tool/demo.png
---

Image size reducer is much needed tool these days because most site wants us to upload various documents in the form of image but with the size limit. Modern days camera gives us image with size in megabytes but the server or site we need to upload the site has size limitation. There are sites like https://www.reduceimages.com to reduce size online but many have limitations like some have usage limits per day, some have too many ads and some have unfriendly usages. With the mindset to create own image size reducer which could have the features as I want, I am creating one using OpenCV and Streamlit.
The version of streamlit is 1.12.0 on this blog. Below is the demo of an working app we are going to build.


<figure>
<video src = "{{site.url}}/assets/image_tool/image_reducer.webm" width="100%" controls> </video>
<figcaption style = "text-align:left; font-style:italic">Image Size Reducer</figcaption>
</figure> 


## Project Structure
Since the project is currently in the beginning, its not a bad idea to follow the following project structure.

![]({{site.url}}/assets/image_tool/proj_str.png)

### Config File
Config file will be used to prepare the configuration of our app. In the future, we might want to change this because changing a config from web app is even easier for an admin. For now, config file will look like below:

```json
{
    "version": "0.0.1",
    "logging": {
        "level": "DEBUG",
        "console_log": true
    },
    "allowed_modes":{
        "Image Size Reducer":{
            "extensions":"png,jpeg,jpg"
        }
    },
    "execution_mode":
    {
        "mode":"dev"
    }

}
```

In above config file, we have specified version, logging and allowed_modes. We have currently set up Image Size Reducer as an allowed mode but could add more later.

### `variables.py` File

This file is used to create global variables by reading config file above so that we dont have to worry about global variables later. Here, we will simply read the file and assign values into variables.

```python
"""Module to define global variables.
"""
import json
import inspect
import os


class Variables:
    """A class to read config file and hold variables.
    """
    
    # Reading paths
    curr_dir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    root_dir = os.path.dirname(curr_dir)
    conf_dir = os.path.join(root_dir, "config")

    # Reading Config file path
    config_file_path = os.path.join(conf_dir, "config.json")
    path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Reading and loading config file
    with open(config_file_path, "r") as file:
        config = json.load(file)

    # logging variables
    logging = config["logging"]
    level = logging["level"]
    console_log = logging["console_log"]
    version = config["version"]

    # allowed modes
    allowed_modes_dict = config["allowed_modes"]
    allowed_modes = list(allowed_modes_dict.keys())

    def __getitem__(self, key):
        """Get the value from key."""
        return self.config[key]


try:
    var = Variables()
except Exception as err:
    raise err
```


### Application File: `app.py`
This file will hold all the codes that will handle the UI and flow of the image reducing. Lets first import necessary modules.

* Import streamlit, opencv, numpy, os, var, Image from PIL to read byte image data.
* Set streamlit app's layout wide.
* Prepare sidebar from where we will select modes.
* Prepare first mode, `Image Size Reducer`.
* If selected, show its selected.
* Select an image with predefined extensions and set it in uploaded file. 
* If its not null, then read it and show the array image.
* If no mode is selected, show it too.

```python
import streamlit as st
import cv2
import numpy as np
import os
from utils.variables import var
from PIL import Image
st.set_page_config(layout="wide")

sidebar = st.sidebar
sidebar.markdown("## Modes ")

size_reducer = sidebar.checkbox("Image Size Reducer")

if size_reducer:
    st.markdown("## Selected Size Reducer")
    exts = var.allowed_modes_dict["Image Size Reducer"]["extensions"].split(",")
    uploaded_file = st.file_uploader(f"Select file: {exts}", type=exts)
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = np.array(img)
        st.image(img, use_column_width=True)


else:
    st.markdown("## No Mode Selected!!")



```

## Developing Image Size Reducer

### Store Temp Image and Remove Old Image
Lets use opencv to reduce image size first. Lets `import time` as well because we want to save images with the timestamp in file name.

```python
if size_reducer:
    st.markdown("## Selected Size Reducer")
    exts = var.allowed_modes_dict["Image Size Reducer"]["extensions"].split(",")
    uploaded_file = st.file_uploader(f"Select file: {exts}", type=exts)
    if uploaded_file is not None:
        fname = f"data/{int(time.time())}."+uploaded_file.name.split(".")[-1]

        img = Image.open(uploaded_file).convert("RGB")        
        with open(fname, "wb") as f:
            f.write(uploaded_file.getbuffer())
            for f in os.listdir("data"):
                ts = float(f.split(".")[0])
                try:
                    if time.time()-ts > 120:
                        os.remove("data/"+f)
                except:
                    pass
```

In above code,
* We prepared a file name to store it in our `data` folder.
* We read the image and converted it into RGB because by default, we will have `alpha` in Image read by PIL. And we are only working with colorspace and image size so lets ignore it.
* Next, create a image file using the byte data in `uploaded_file` by writing in a file pointer created with `wb` in `fname`.
* Also, the files will get created everytime widget's states are updated so there will be a lot of file withing a minute inside our data folder, we need to remove them if its been more than 3 minutes since it has been created. We use the timestamp that we attached in the filename to find the minutes since its created.

### Widgets for New Height/Width

Now that we have stored the file as it was uploaded, its time for us to show its default size, shape and image if needed. Also show the new height and width that we want to have.

```python
        img = np.asarray(img)
        H,W,_=img.shape        
        show_image = st.checkbox("Show Image")
        if show_image:
            st.image(img, use_column_width=True)
        
        
        st.markdown(f"""Original Dimension of the image is: {H,W}. \\
                        Original Size of the image is: {os.path.getsize(fname)/1024}kbs \\
                        Please Select H and W.""")
        
        cols = st.columns(2)
        h = cols[0].number_input("Height", min_value=1, value=int(H))
        w = cols[1].number_input("Width",min_value=1, value=int(W))
```

In above code,
* We showed the dimension of image, size of image in KBs.
* Created two widgets for Height and Width. **Height of Image is related to Y-axis and Width is related to X-axis.** And the `H,W,_ = img.shape` gives the number of rows, cols.

### Resize Image and Save it!

Lets use the values from the widgets above to resize image size and save it.

```python
        if st.button("Reduce size!!"):
            nimg = cv2.resize(img, (int(w), int(H)), interpolation=cv2.INTER_AREA)
            nfname = fname.replace("data/", "data/temp_")
            
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

            if show_image:
                st.image(img, use_column_width=True)


            cv2.imwrite(nfname, nimg)
            st.markdown(f"New file size: {os.path.getsize(nfname)/1024} kbs")
```

### Download Resized Image
Lets download the resized image.

```python
            with open(nfname, "rb") as fp:
                dbtn = st.download_button(label="Download image file.", data=fp,
                            file_name=nfname.split("/")[-1], mime="image/png")
                if dbtn:
                    st.markdown("Downloaded!!!!")
```

In above code, we tried to open recently saved resized image in byte format and then put it in `download_button`'s data. Once its downloaded, `Downloaded!!!!` is shown.

## Full Code

### `app.py`

```python
import streamlit as st
import cv2
import numpy as np
import os
from utils.variables import var
from PIL import Image
import time

st.set_page_config(layout="wide")

sidebar = st.sidebar
sidebar.markdown("## Modes ")

size_reducer = sidebar.checkbox("Image Size Reducer")

if size_reducer:
    st.markdown("## Selected Size Reducer")
    exts = var.allowed_modes_dict["Image Size Reducer"]["extensions"].split(",")
    uploaded_file = st.file_uploader(f"Select file: {exts}", type=exts)
    if uploaded_file is not None:
        fname = f"data/{int(time.time())}."+uploaded_file.name.split(".")[-1]

        img = Image.open(uploaded_file).convert("RGB")        
        with open(fname, "wb") as f:
            f.write(uploaded_file.getbuffer())
            for f in os.listdir("data"):
                ts = float(f.split(".")[0])
                try:
                    if time.time()-ts > 120:
                        os.remove("data/"+f)
                except:
                    pass

        img = np.asarray(img)
        H,W,_=img.shape        
        show_image = st.checkbox("Show Image")
        if show_image:
            st.image(img, use_column_width=True)
        
        
        st.markdown(f"""Original Dimension of the image is: {H,W}. \\
                        Original Size of the image is: {os.path.getsize(fname)/1024}kbs \\
                        Please Select H and W.""")
        
        cols = st.columns(2)
        h = cols[0].number_input("Height", min_value=1, value=int(H))
        w = cols[1].number_input("Width",min_value=1, value=int(W))

        if st.button("Reduce size!!"):
            nimg = cv2.resize(img, (int(w), int(H)), interpolation=cv2.INTER_AREA)
            nfname = fname.replace("data/", "data/temp_")
            
            nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

            if show_image:
                st.image(img, use_column_width=True)


            cv2.imwrite(nfname, nimg)
            st.markdown(f"New file size: {os.path.getsize(nfname)/1024} kbs")

            with open(nfname, "rb") as fp:
                dbtn = st.download_button(label="Download image file.", data=fp,
                            file_name=nfname.split("/")[-1], mime="image/png")
                if dbtn:
                    st.markdown("Downloaded!!!!")
else:
    st.markdown("## No Mode Selected!!")



```

Please follow [this link](https://github.com/q-viper/Image-Processing-Web-Tool/releases/tag/V0.0.1) for the full codes.

This is all for now in this blog and there are a lot to come in this blog soon. I will add different features like color changing, convolving, edge detecting and many cool Image Processing algorithms. Stay Tuned!!


```python

```
