import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from io import BytesIO
import sys
import pandas as pd


sys.dont_write_bytecode = True
sys.tracebacklimit = 0
from modules import *

st.set_page_config(
    page_title="BobTextureAnalysis",
    page_icon="🔬",
    layout="wide",
)

col1, col2 = st.columns(2)

with col1:
    # title for form
    st.markdown('test')

    msg = "Upload a 2D image to be analyzed. Works best with images smaller than 600x600 pixels."

    # User input:
    # Image to analyze
    # Checkbox to invert image
    # slider for local sigma

    uploaded_file = st.file_uploader(msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False, label_visibility='collapsed')

    invert = st.checkbox(label="Invert image?")

    inner_sigma = st.slider(min_value=1, max_value=20, step=1, label="Local sigma", help="Smaller values will emphasize higher frequecy detail, while larger values will focus on larger detail. Find what looks cool!")
    
    
    # Adding user input to st.session_state

    # Image
    if "opencv_image" not in st.session_state:
        st.session_state.opencv_image = None
    
    if uploaded_file:
        # read file as bytes, then decode into opencv image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        st.session_state.opencv_image = opencv_image
        st.write("Image uploaded successfully!")
    

    # Invert?
    if "invert_image" not in st.session_state:
        st.session_state.invert_image = None
    
    
    
    st.session_state.invert_image = invert



    # Local sigma
    if "inner_sigma" not in st.session_state:
        st.session_state.inner_sigma = None
    
    if inner_sigma:
        st.session_state.inner_sigma = inner_sigma
    
    phase = st.slider("Angle phase shift (degrees)", min_value=0, max_value=360, step=1)

    if "angle_phase_shift" not in st.session_state:
            st.session_state.angle_phase_shift = None

    st.session_state.angle_phase_shift = phase

with col2:
    if st.session_state.opencv_image is not None:
        raw_image_gray = color.rgb2gray(st.session_state.opencv_image)

        if st.session_state.invert_image:
            raw_image_gray = 1 - raw_image_gray
        
        coherence, two_phi = coh_ang_calc(raw_image_gray, sigma_inner=st.session_state.inner_sigma, gradient_mode='sobel')
        if "coh_ang" not in st.session_state:
            st.session_state.coh_ang = None
        st.session_state.coh_ang = (coherence, two_phi)

        all_img = orient_hsv(raw_image_gray, coherence, two_phi, mode='all', angle_phase=st.session_state.angle_phase_shift)
        coh_img = orient_hsv(raw_image_gray, coherence, two_phi, mode='coherence')
        ang_img = orient_hsv(raw_image_gray, coherence, two_phi, mode='angle', angle_phase=st.session_state.angle_phase_shift)
        ang_img_bw = orient_hsv(raw_image_gray, coherence, two_phi, mode="angle_bw", angle_phase=st.session_state.angle_phase_shift)

        st.image([raw_image_gray, all_img, coh_img, ang_img, ang_img_bw], caption=["raw grayscale image", "Coherence, angle, image: angle is hue, coherence is saturation, brightness is original image", "Coherence only", "Angle and coherence", "Angle only, grayscale"])
    else:
        st.write("No image uploaded yet")
