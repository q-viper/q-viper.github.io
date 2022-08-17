---
title:  "Building Image Merger Web Tool In Python"
date:   2022-08-21 01:29:17 +0545
categories:
    - Image Processing
    - Web App
tags:
    - Image Merger
header:
  teaser: assets/image_tool/hmargin.png
---

Image Merger tool is also useful just like the image size reducer tool. Lets take an example where is it needed, suppose we are about to fill some application where we have to submit our passport copy. We need the first and last page's copy and we could either make a single image by combining the two images or print them one after another. Clearly, having both page in a single image is a win. But to do so, we have to use some tools like MS Paint, online tools. Both are time consuming because MS paint needs adjusments of image sizes and online tools have too many ads, usage limitations and so on. But why not create our own image merger tool? 

![]({{site.url}}/assets/image_tool/merger.png)

This blog is the continuation of the previous blog [Making own Image Size Reducer in Python](https://dataqoil.com/2022/08/14/building-image-size-reducer-tool-in-python/). Again, we will use the code from that project and continue to add feature.

In above figure, there are 4 inputs,
* Show image
* Merge Horizonatlly
* Margin Between
* Select Files

All of these takes part in merging.


## Adding App's Title and Icon
In the last part, we only focused on the function of the app and did not do anythin to change the title and icon. But lets do that now.

```python
st.set_page_config(layout='wide', page_title="Image Online Tool",
                    page_icon="data/icon.png",
                    menu_items={
                        'About': "# Image Online Tool!"
                })
```

The `icon.png` in above code is created by myself and its like below:

![]({{site.url}}/assets/image_tool/icon.png)

## Making `app.py`Cleaner
In the previous code, all the codes to reduce image size was inside `app.py` but lets move those codes to make app.py more cleaner. We will move the codes inside `utils/utils.py` file. The `app.py` should look like below:

```python
"""
Main app file.
"""
from email.mime import image
import streamlit as st
from utils.variables import var
from utils.utils import *

st.set_page_config(layout='wide', page_title="Image Online Tool",
                    page_icon="assets/icon.png",
                    menu_items={
                        'About': "# Image Online Tool!"
                })

sidebar = st.sidebar
sidebar.markdown("## Modes ")

mode_size_reducer = sidebar.checkbox("Image Size Reducer")
mode_image_merger = sidebar.checkbox("Image Merger")

remove_old()

if mode_size_reducer:
    st.markdown("## Selected Size Reducer")
    size_reducer(st)
if mode_image_merger:
    st.markdown("## Selected Image Merger")
    image_merger(st)
```

The functions `size_reducer` and `image_merger` are defined inside `utils/utils.py`.

The function `remove_old` is new function which is used to remove old files to free up the space. This function is also defined inside `utils.py`

```python
def remove_old():
    for f in os.listdir("data"):
        if "temp" in f:        
            ts = float(f.split(".")[0].split("_")[-1])
        else:
            ts = float(f.split(".")[0])
        try:
            if time.time()-ts > 60:
                os.remove("data/"+f)
        except:
            pass
```

## `size_reducer` Function
As it is. But we will accept the instance of streamlit as `st` from the `app.py`. This way we could work with main streamlit instance and app.

```python
def size_reducer(st):
    exts = var.allowed_modes_dict["Image Size Reducer"]["extensions"].split(",")
    uploaded_file = st.file_uploader(f"Select file: {exts}", type=exts)
    if uploaded_file is not None:
        fname = f"data/{int(time.time())}."+uploaded_file.name.split(".")[-1]

        img = Image.open(uploaded_file).convert("RGB")        
        with open(fname, "wb") as f:
            f.write(uploaded_file.getbuffer())
            

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
            nimg = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
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
```

## `image_merger` Function
Now comes the part to merge images. Please refer to the comment above the code line for the explanation.

```python
def image_merger(st):
    exts = var.allowed_modes_dict["Image Merger"]["extensions"].split(",")
    
    # do we want to show image?
    show_image = st.checkbox("Show Image")
    
    # merge horizontally or vertically? True for horizonatl merge.
    merge_horizontally = st.checkbox("Merge Horizontally")
    
    # margin between images
    margin = st.number_input("Margin between two images.", value=10)

    all_imgs = {}
    uploaded_file = st.file_uploader(f"Select file: {exts}", type=exts, accept_multiple_files=True)
    files = []

    if uploaded_file is not None:
        
        # for new height and width of merged image. It will be average!
        nh=0
        nw=0
        for uf in uploaded_file:
            fname = f"data/{int(time.time())}."+uf.name #.split(".")[-1]
            
            bd = uf.read()

            img = Image.open(io.BytesIO(bd)).convert("RGB")        
            with open(fname, "wb") as f:
                f.write(uf.getbuffer())
                
            img = np.asarray(img)
            H,W,_=img.shape        
            
            # adding new height and width
            nh+=H
            nw+=W

            if show_image:
                st.markdown(f"### {uf.name}")
                st.image(img, use_column_width=True)
            
            # appending image name in the list.
            files.append(fname)
            # all_imgs[fname] = img
    
     # if image files list is not empty,
    if len(files)>0:
        # st.write(files)
        
        # find average width and height
        fs=len(uploaded_file)
        nh = int(nh/fs)
        nw = int(nw/fs)
        
        # prepare margin as white image of shape wrt. horizontal or not
        if merge_horizontally:
            mr = np.zeros((nh, int(margin), 3))+255.0
        else:
            mr = np.zeros((int(margin), nw, 3))+255.0
        
        # new image where merged image will be present
        nimg = None
        
        # looping through image names
        for file in files:
            
            # read and reverse the colorspace from BGR to RGB
            img = cv2.imread(file)
            img = img[:,:,::-1]
            
            # resize into new average size
            img = cv2.resize(img, (nw, nh))
            # st.image(img)
            
            # stack image into nimg. If nimg is empty, set it to img
            # st.image(mr)
            if nimg is None:
                nimg = img.copy()
                nimg = nimg.astype(np.uint8)
                # st.image(nimg, use_column_width=True)
            else:
                # do hstack for horizonatl and vstack for vertical merge
                if merge_horizontally:
                    nimg = np.hstack([nimg, mr, img])
                else:
                    nimg = np.vstack([nimg, mr, img])
                
                # assign type of image
                nimg = nimg.astype(np.uint8)
                # st.image(nimg, use_column_width=True)
        
        # show merged image
        st.markdown("### Merged Image")
        st.image(nimg, use_column_width=True) 
        
        # save new image as temp
        nfname=f"data/temp_{int(time.time())}.png"
        cv2.imwrite(nfname, nimg[:,:,::-1])
        
        # open new image as byte and assign it into download button
        with open(nfname, "rb") as fp:
            dbtn = st.download_button(label="Download image file.", data=fp,
                        file_name=nfname.split("/")[-1], mime="image/png")
            if dbtn:
                st.markdown("Downloaded!!!!")

```


The horizontal merger looks like below:
![]({{site.url}}/assets/image_tool/hmerger.png)

The margin also takes a huge part in merging.
![]({{site.url}}/assets/image_tool/hmargin.png)


The source code is available at [GitHub link](https://github.com/q-viper/Image-Processing-Web-Tool/tree/V0.0.2).