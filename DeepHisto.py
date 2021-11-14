import fastai
import streamlit as st
from fastai.basics import *
from fastai.vision import *
import streamlit.components.v1 as components

import PIL
#import torch
#import torchvision

import cv2
from PIL import Image
import numpy as np
import time


# st.set_page_config(layout="wide")


import os
import io

st.title('Cartilage scoring')





def main():


    menu = ["Home","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.sidebar.subheader("Home")

    elif choice == "Login":
        st.sidebar.subheader("Login Section is under development")


    elif choice == "SignUp":
        st.sidebar.subheader("Signup section is under development")




if __name__=='__main__':
    main()


st.sidebar.subheader('Theme')
lt = st.sidebar.checkbox('Light theme')

st.sidebar.subheader('Process')
proc = st.sidebar.checkbox('Include preprocessing on the image')
sample = st.sidebar.selectbox('Sample:',('Human sample',''))
anal = st.sidebar.selectbox('Analyze:',('Integrity', 'Mankin Score', 'OARSI Score'))

s = f"""
<style>
    .block-container {{padding:10px !important; }}
    
    .stCheckbox {{width:150px !important;}}
    .stButton {{width:200px !important;}}
    .css-1nafuxo {{display: none}}
    #MainMenu {{display: none}}
    .e16nr0p30 {{color: 616161;}}
    p {{color: #949494;}}
    h3 {{color: #616161;}}
    .effi0qh0 {{color: #949494;}}
    .etr89bj0 {{width: 50px;}}
</style>
"""
st.markdown(s, unsafe_allow_html=True)

tut = 0
tutbut = st.button('Show tutorial')
tutheight = 0
tutbutclose = st.button('Hide tutorial')

if tutbut and tut==0:
    tutheight = 500
    tut = 1
    # print(tutheight)

if tutbutclose and tut==1:
    tutheight = 0
    tut = 0
    # print(tutheight)

components.html(
"""
    <style>
        .accordion {
          background-color: #eee;
          color: #444;
          cursor: pointer;
          padding: 18px;
          width: 100%;
          border: none;
          text-align: left;
          outline: none;
          font-size: 15px;
          transition: 0.4s;
        }
        
        .active, .accordion:hover {
          background-color: #ccc; 
        }
        
        .panel {
          padding: 0 18px;
          background-color: rgb(150, 150, 150);
          overflow: hidden;
          transition: 1s;
          height:0px;

        }
    </style>
</head>
<body>
    
        
        
<video style="margin-top:10px; margin-left:20px;" width="800" height="400" controls>
<source src="https://share.streamlit.io/soroushoskouei/deephistology/tut.mp4" type="video/mp4">
</video>


""", height = tutheight
)



st.write("Please upload your image for evaluation")



stLight = f"""
<style>
    .block-container {{background-color: beige}}
    .ek41t0m0 {{color: black;}}
    .css-a39chc {{background-color: #343434;}}
    .css-10trblm {{color: Black}}
    .css-j8zjtb {{color: Black; font:Courier New', Courier, monospace !important;}}

<style>
    
"""

stDark = f"""
<style>
    .block-container {{background-color: "#343434"}}

<style>
    
"""
kkk = 0

if lt and kkk == 0:
    st.markdown(stLight, unsafe_allow_html=True)
    txtclr = 'black'
    kkk = 1
else:
    st.markdown(stDark, unsafe_allow_html=True)
    txtclr = 'white'
    kkk = 0






if anal=='Integrity':
    model1 = load_learner('./class1/')
    model2 = load_learner('./class2/')
elif anal=='Mankin Score':
    model = load_learner('./mankin/')
elif anal=='OARSI Score':
    model = load_learner('./oarsi/')



tfms = get_transforms()

thefname = ''


#================================ reduce ==========================================


def reduceit(img,size):
    img = np.array(img)
    
    scale_percent = size # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

#================================= Horiz ==========================================

def horiz(img):
    img = np.array(img)
    OriginalDim = img.shape
    
    scale_percent = 8 # percent of original size
    width = int(img.shape[1]*scale_percent / 100)
    height = int(img.shape[0]*scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    imgr = Image.fromarray(resized)
    imggray = imgr.convert('LA')

    imgmat = np.array(list(imggray.getdata(band=0)), float)
    imgmat.shape = (imggray.size[1], imggray.size[0])
    imgmat = np.matrix(imgmat)




    kernel = np.ones((2,2), np.uint8)
    imagee = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    imagee = cv2.GaussianBlur(imagee, (7,7), 7)
    imagee = cv2.medianBlur(imagee, 5)

    ret, thresh = cv2.threshold(imagee, 130, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

    thresh = cv2.dilate(thresh, kernel, iterations=9)

    contours, a = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2. boxPoints(rect)
        box = np.int0(box)
        imgb = cv2.drawContours(imagee, [box], 0 , (0,0,255), 1) 

    imgr = Image.fromarray(img)
    # print(names, rect[2])
    if rect[2]>45 and rect[2]<80:
        imgr2 = imgr.rotate(45 - rect[2])
    elif rect[2]>80:
        imgr2 = imgr.rotate(0)
    else:
        imgr2 = imgr.rotate(rect[2])


    #deleting blacks
    img2 = np.array(imgr2)
    scale_percent = 15 # percent of original size
    width2 = int(img2.shape[1]*scale_percent / 100)
    height2 = int(img2.shape[0]*scale_percent / 100)
    dim2 = (width2, height2)
    
    # resize image
    resized2 = cv2.resize(img2, dim2, interpolation = cv2.INTER_AREA)

    for i in range(0,height2, 1):
        for j in range(0,width2, 1):
            if resized2[i,j][0] == 0 and resized2[i,j][1] == 0 and resized2[i,j][2] == 0:
                resized2[i,j] = [255,255,255]
                                                                                      
    return resized2
    #cv.imwrite('test4/'+names[0:-3]+'rect.jpg', imgb)



#================================= Cropping ========================================

def cr1(img):
    img = np.array(img)
    imgr = Image.fromarray(img)
    imggray = imgr.convert('LA')

    imgmat = np.array(list(imggray.getdata(band=0)), float)
    imgmat.shape = (imggray.size[1], imggray.size[0])
    imgmat = np.matrix(imgmat)

    # SVD
    U, sigma, V = np.linalg.svd(imgmat)

    # Manipulate SVD
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if U[i,j] > 0:
                U[i,j] = -1
    reconstimg = np.matrix(U[:, :2]) * np.diag(sigma[:2]) * np.matrix(V[:2, :])
    k = 10
    a = []
    for i in reconstimg[10:len(reconstimg[:,5])-10,5]:
        if i[0,0] < 20:
            a.append(k)
        k = k + 1
    
    width, height = imgr.size
    
    if len(a) != 0:
        # Setting the points for cropped image

        top = min(a)
        bottom = max(a)

        left = 0
        top = top-10
        right = width
        bottom = height

        # Cropped image of above dimension
        im1 = img[top:bottom, left:right]
        return im1

    else:

        reconstimg = np.matrix(U[:, :1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])
        k = 10
        a = []
        for i in reconstimg[10:len(reconstimg[:,5])-10,5]:
            if i[0,0] < 210:
                a.append(k)
            k = k + 1

        width, height = imgr.size
    
        if len(a) != 0:

            top = min(a)
            bottom = max(a)


            left = 0
            top = top-10
            right = width
            bottom = height

            # Cropped image of above dimension
            im1 = img[top:bottom, left:right]
            return im1

        else:
            return img







#================================= Window1 ========================================

# def wndw1(img):
#     rtts = []
#     img = np.array(img)
#     OriginalDim = img.shape
#     height = OriginalDim[0]
#     width = OriginalDim[1]
#     #print('file: ',names)

#     wl = width/3
#     n = 4
#     a = (n*wl - width)/(n-1)

#     def addition(n):
#         return n + wl

#     left = [0, wl - a, 2*wl - 2*a, 3*wl - 3*a]
#     right = list(map(addition, left))



#     top = 0
#     bottom = height

#     imgg = Image.fromarray(img)
 
# # Cropped image of above dimension
# # (It will not change original image)
#     im0 = imgg.crop((left[0], top, right[0], bottom))
#     im1 = imgg.crop((left[1], top, right[1], bottom))
#     im2 = imgg.crop((left[2], top, right[2], bottom))
#     im3 = imgg.crop((left[3], top, right[3], bottom))

#     #imgr = Image.fromarray(img)
#     rtts.append(im0)
#     rtts.append(im1)
#     rtts.append(im2)
#     rtts.append(im3)
#     #deleting blacks

#     return rtts







#================================= Bone Delete ======================================

def BoneDelete(img):
    
    # img = cv2.imread('static/'+thefname, cv2.IMREAD_UNCHANGED)
    img = np.array(img)
    OriginalDim = img.shape
    # print('Original Dimensions : ',OriginalDim)
    
    #scale_percent = 15 # percent of original size
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    #dim = (width, height)
    ## resize image
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)



    ## img = Image.open('r1.jpg')
    #imgr = Image.fromarray(resized)
    imgr = Image.fromarray(img)
    imggray = imgr.convert('LA')

    imgmat = np.array(list(imggray.getdata(band=0)), float)
    imgmat.shape = (imggray.size[1], imggray.size[0])
    imgmat = np.matrix(imgmat)
    # plt.figure(figsize=(9,6))
    # plt.imshow(imgmat, cmap='gray');

    # SVD
    U, sigma, V = np.linalg.svd(imgmat)

    # Manipulate SVD
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            if U[i,j] > 0:
                U[i,j] = -1
    reconstimg = np.matrix(U[:, :2]) * np.diag(sigma[:2]) * np.matrix(V[:2, :])
    k = 10
    a = []
    for i in reconstimg[10:len(reconstimg[:,5])-10,5]:
        if i[0,0] < 20:
            a.append(k)
        k = k + 1
    
    width, height = imgr.size
    
    if len(a) != 0:
        # Setting the points for cropped image
        # left = 5
        top = min(a)
        bottom = max(a)
        # ge of above dimension
        # (It will not change original image)

        # im1 = imgr.crop((0, top, width, bottom))
        # im1
        IMREAD_COLOR = img

        for i in range(0,height, 1):
            for j in range(0,width, 1):
                if i>round(top + (bottom-top)/2) and IMREAD_COLOR[i,j][2] > 170 and IMREAD_COLOR[i,j][0] > 180:
                            IMREAD_COLOR[i,j] = [255,255,255]


        for i in range(0, height, 1):
                for j in range(0,width, 1):
                    if IMREAD_COLOR[i,j][0] > 210 and IMREAD_COLOR[i,j][1] > 210 and IMREAD_COLOR[i,j][2] > 210:
                        IMREAD_COLOR[i,j] = [255,255,255]

        for i in range(round(top+(bottom-top)/2),height-6, 1):
                for j in range(0,width, 1):
                    if IMREAD_COLOR[i-6,j][0] > 210 and IMREAD_COLOR[i-6,j][1] > 210 and IMREAD_COLOR[i-6,j][2] > 210 and IMREAD_COLOR[i+6,j][0] > 210 and IMREAD_COLOR[i+6,j][1] > 210 and IMREAD_COLOR[i+6,j][2] > 210:
                        IMREAD_COLOR[i,j] = [255,255,255]

        for i in range(0, height, 1):
                for j in range(0,width, 1):
                    if IMREAD_COLOR[i,j][1] == IMREAD_COLOR[i,j][2]:
                        IMREAD_COLOR[i,j] = [255,255,255]

        for i in range(0,height-3, 1):
                for j in range(0,width, 1):
                    if IMREAD_COLOR[i-3,j][0] > 210 and IMREAD_COLOR[i-3,j][1] > 210 and IMREAD_COLOR[i-3,j][2] > 210 and IMREAD_COLOR[i+3,j][0] > 210 and IMREAD_COLOR[i+3,j][1] > 210 and IMREAD_COLOR[i+3,j][2] > 210:
                        IMREAD_COLOR[i,j] = [255,255,255]

        for i in range(0,height, 1):
                for j in range(2,width-2, 1):
                    if IMREAD_COLOR[i,j-2][0] > 210 and IMREAD_COLOR[i,j-2][1] > 210 and IMREAD_COLOR[i,j-2][2] > 210 and IMREAD_COLOR[i,j+2][0] > 210 and IMREAD_COLOR[i,j+2][1] > 210 and IMREAD_COLOR[i,j+2][2] > 210:
                        IMREAD_COLOR[i,j] = [255,255,255]
            
        #cv2.imshow('BDR/'+kk,IMREAD_COLOR)
        # cv2.imwrite('static/'+thefname, IMREAD_COLOR)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return IMREAD_COLOR

    else:

        reconstimg = np.matrix(U[:, :1]) * np.diag(sigma[:1]) * np.matrix(V[:1, :])
        k = 10
        a = []
        for i in reconstimg[10:len(reconstimg[:,5])-10,5]:
            if i[0,0] < 210:
                a.append(k)
            k = k + 1
    
        width, height = imgr.size
    
        if len(a) != 0:
            # Setting the points for cropped image
            # left = 5
            top = min(a)
            bottom = max(a)
            # ge of above dimension
            # (It will not change original image)

            # im1 = imgr.crop((0, top, width, bottom))
            # im1

            IMREAD_COLOR = img

            for i in range(0,height, 1):
                    for j in range(0,width, 1):
                        if i>round(top + (bottom-top)/2) and IMREAD_COLOR[i,j][2] > 170 and IMREAD_COLOR[i,j][0] > 180:
                            IMREAD_COLOR[i,j] = [255,255,255]




            for i in range(0, height, 1):
                    for j in range(0,width, 1):
                        if IMREAD_COLOR[i,j][0] > 210 and IMREAD_COLOR[i,j][1] > 210 and IMREAD_COLOR[i,j][2] > 210:
                            IMREAD_COLOR[i,j] = [255,255,255]

            for i in range(round(top + (bottom-top)/2),height-6, 1):
                    for j in range(0,width, 1):
                        if IMREAD_COLOR[i-6,j][0] > 210 and IMREAD_COLOR[i-6,j][1] > 210 and IMREAD_COLOR[i-6,j][2] > 210 and IMREAD_COLOR[i+6,j][0] > 210 and IMREAD_COLOR[i+6,j][1] > 210 and IMREAD_COLOR[i+6,j][2] > 210:
                            IMREAD_COLOR[i,j] = [255,255,255]


            for i in range(0,height-3, 1):
                    for j in range(0,width, 1):
                        if IMREAD_COLOR[i-3,j][0] > 210 and IMREAD_COLOR[i-3,j][1] > 210 and IMREAD_COLOR[i-3,j][2] > 210 and IMREAD_COLOR[i+3,j][0] > 210 and IMREAD_COLOR[i+3,j][1] > 210 and IMREAD_COLOR[i+3,j][2] > 210:
                            IMREAD_COLOR[i,j] = [255,255,255]

            for i in range(0,height, 1):
                    for j in range(2,width-2, 1):
                        if IMREAD_COLOR[i,j-2][0] > 210 and IMREAD_COLOR[i,j-2][1] > 210 and IMREAD_COLOR[i,j-2][2] > 210 and IMREAD_COLOR[i,j+2][0] > 210 and IMREAD_COLOR[i,j+2][1] > 210 and IMREAD_COLOR[i,j+2][2] > 210:
                            IMREAD_COLOR[i,j] = [255,255,255]

            for i in range(0, height, 1):
                    for j in range(0,width, 1):
                        if IMREAD_COLOR[i,j][1] == IMREAD_COLOR[i,j][2]:
                            IMREAD_COLOR[i,j] = [255,255,255]

            #cv.imshow('BDR/'+kk,IMREAD_COLOR)
            # cv2.imwrite('static/'+thefname, IMREAD_COLOR)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return IMREAD_COLOR

        else:
            # print(thefname,' Does not work')
            # cv2.imwrite('static/'+thefname, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return img



#========================================== predict ====================================================
# def predict_label(img_path):
#     i = open_image(img_path)
#     i = i.apply_tfms(tfms[0], size=224)
#     # i = image.img_to_array(i)/255.0
#     # i = i.reshape(1, 224,224,3)
#     # i = PIL.Image.fromarray(i)
#     p = model.predict(i)
#     # return dic[p[0]]
#     return(p[0])

# st.write(predict_label('./uploads/05.jpg'))


uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=224)
    slider_output =  st.slider('Choose the reduce percentage', 0, 100, 50)

    if st.button('Predict'):
        if proc:
            gif_path = './p.gif'
            gif_runner = st.image(gif_path)
            # print(slider_output)
            image = reduceit(image,slider_output)
            image = horiz(image)
            st.image(image, caption='Rotated Image.', width=224)
            
            i = BoneDelete(image)
            st.image(i, caption='Processed Image', width=50)

            from fastai.vision import *
            i = Image(pil2tensor(i, dtype=np.float32).div_(255))
            # st.write(type(i))

            i = i.apply_tfms(tfms[0], size=224)
            
            st.write("")

#=================integrity + preprocess==============================================================
            if anal=='Integrity':
                st.write("Classifying...")
                label = model1.predict(i)
                probebs = label[2].detach().numpy()
                st.write('Probabilities: ')
                st.write('Mild: ',probebs[0])
                st.write('Not Mild: ',probebs[1])

                if probebs[0]>=0.8 or probebs[1]>=0.8:
                    original_title = '<p style="color:Green">Predicted: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)
                elif probebs[0]<0.8 or probebs[1]<0.8:
                    original_title = '<p style="color:Yellow">Predicted: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)  
                gif_runner.empty()

                if str(label[0]) == 'notMild':
                    st.write("Classifying...")
                    label = model2.predict(i)
                    probebs = label[2].detach().numpy()
                    st.write('Probabilities: ')
                    st.write('Advanced: ',probebs[0])
                    st.write('Moderate: ',probebs[1])

                    if probebs[0]>=0.8 or probebs[1]>=0.8:
                        original_title = '<p style="color:Green">Predicted: </p>' + str(label[0])
                        st.markdown(original_title, unsafe_allow_html=True)
                    elif probebs[0]<0.8 or probebs[1]<0.8:
                        original_title = '<p style="color:Yellow">Predicted: </p>' + str(label[0])
                        st.markdown(original_title, unsafe_allow_html=True)  
                    gif_runner.empty()    

#=================mankin + preprocess==============================================================
            elif anal=='Mankin Score':

                label = model.predict(i)
                probebs = label[2].detach().numpy()

                if probebs[0]>=0.8:
                    original_title = '<p style="color:Green">Predicted Score: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)
                elif probebs[0]<0.8:
                    original_title = '<p style="color:Yellow">Predicted Score: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)  
                gif_runner.empty()

#=================Oarsi + preprocess==============================================================
            elif anal=='OARSI Score':

                label = model.predict(i)
                probebs = label[2].detach().numpy()

                if probebs[0]>=0.8:
                    original_title = '<p style="color:Green">Predicted Score: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)
                elif probebs[0]<0.8:
                    original_title = '<p style="color:Yellow">Predicted Score: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)  
                gif_runner.empty()


#=================integrity + NO preprocess==============================================================

        else:
            i = open_image(uploaded_file)
            i = i.apply_tfms(tfms[0], size=224)
            st.write("")
            
            if anal=='Integrity':
                st.write("Classifying...")
                label = model1.predict(i)
                probebs = label[2].detach().numpy()
                st.write('Probabilities: ')
                st.write('Mild: ',probebs[0])
                st.write('Not Mild: ',probebs[1])

                if probebs[0]>=0.8 or probebs[1]>=0.8:
                    original_title = '<p style="color:Green">Predicted: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)
                elif probebs[0]<0.8 or probebs[1]<0.8:
                    original_title = '<p style="color:Yellow">Predicted: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)  
                gif_runner.empty()

                if str(label[0]) == 'notMild':
                    st.write("Classifying...")
                    label = model2.predict(i)
                    probebs = label[2].detach().numpy()
                    st.write('Probabilities: ')
                    st.write('Advanced: ',probebs[0])
                    st.write('Moderate: ',probebs[1])

                    if probebs[0]>=0.8 or probebs[1]>=0.8:
                        original_title = '<p style="color:Green">Predicted: </p>' + str(label[0])
                        st.markdown(original_title, unsafe_allow_html=True)
                    elif probebs[0]<0.8 or probebs[1]<0.8:
                        original_title = '<p style="color:Yellow">Predicted: </p>' + str(label[0])
                        st.markdown(original_title, unsafe_allow_html=True)                

#=================Mankin + No preprocess==============================================================

            elif anal=='Mankin Score':  

                label = model.predict(i)
                probebs = label[2].detach().numpy()

                if probebs[0]>=0.8:
                    original_title = '<p style="color:Green">Predicted Score: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)
                elif probebs[0]<0.8:
                    original_title = '<p style="color:Yellow">Predicted Score: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)   

#=================Oarsi + No preprocess==============================================================

            elif anal=='OARSI Score':  

                label = model.predict(i)
                probebs = label[2].detach().numpy()
 
                if probebs[0]>=0.8:
                    original_title = '<p style="color:Green">Predicted Score: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)
                elif probebs[0]<0.8:
                    original_title = '<p style="color:Yellow">Predicted Score: </p>' + str(label[0])
                    st.markdown(original_title, unsafe_allow_html=True)  





