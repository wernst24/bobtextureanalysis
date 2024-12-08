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

# """
# TODO:
# 0. decompose coherence and angle calculation
# 1. cache intermediate data to reduce memory cost, and intelligently update when options updated
# 2. add options for gradient calculation, to get rid of sobel artifacts (yuck)
# """




col1, col2, col3 = st.columns(3)

with col1:
    with st.form(key='form1', clear_on_submit=False):
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


        st.form_submit_button(label="Analyze image")


        # Local sigma
        if "inner_sigma" not in st.session_state:
            st.session_state.inner_sigma = None
        
        if inner_sigma:
            st.session_state.inner_sigma = inner_sigma

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

with col3:
    phase = st.slider("Angle phase shift (degrees)", min_value=0, max_value=360, step=1)

    if "angle_phase_shift" not in st.session_state:
            st.session_state.angle_phase_shift = None

    st.session_state.angle_phase_shift = phase

    with open("test_semicircles.png", "rb") as f:
        semicircle_read = f.read()
    semicircle_bytes = np.asarray(bytearray(semicircle_read), dtype=np.uint8)
    semicircle_image = color.rgb2gray(cv.imdecode(semicircle_bytes, 1))
    semi_coh, semi_ang = coh_ang_calc(semicircle_image, sigma_inner=st.session_state.inner_sigma, gradient_mode='sobel')
    st.image(orient_hsv(semicircle_image, semi_coh, semi_ang, mode='all', angle_phase=st.session_state.angle_phase_shift), caption="Angles from -pi/2 to pi/2")

    if "coh_ang" not in st.session_state:
            st.session_state.coh_ang = None
    
    if st.session_state.coh_ang:
        coherence, angles = st.session_state.coh_ang
        st.write("Warning: very, very, very messy csv format. Only for the brave")
        with BytesIO() as buffer1:
            np.savetxt(buffer1, coherence, delimiter=",")
            st.download_button("Coherence image as csv", buffer1, file_name="coherence_image", mime="text/csv")
        with BytesIO() as buffer2:
            np.savetxt(buffer2, angles, delimiter=",")
            st.download_button("Angle image as csv", buffer2, file_name="angle_image", mime="text/csv")