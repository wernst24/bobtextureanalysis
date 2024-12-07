import streamlit as st

import numpy as np

import matplotlib.pyplot as plt

import time

from io import BytesIO

import sys

sys.dont_write_bytecode = True

sys.tracebacklimit = 0

from modules import *

col1, col2 = st.columns(2)

with col1:
    with st.form(key='form1', clear_on_submit=False):
        # title for form
        st.markdown('test')

        msg = "Upload a 2D image to be analyzed. Works best with images smaller than 600x600 pixels."
        uploaded_file = st.file_uploader(msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False, label_visibility='collapsed')
            
        chunk_size = st.slider("Downscale factor", min_value=1, max_value=10, step=1)

        if "opencv_image" not in st.session_state:
            st.session_state.opencv_image = None
        
        if uploaded_file:
            # read file as bytes, then decode into opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv.imdecode(file_bytes, 1)
            st.session_state.opencv_image = opencv_image
            st.write("Image uploaded successfully!")

        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = None
        
        if chunk_size:
            st.session_state.chunk_size = chunk_size

        st.form_submit_button(label="Analyze image")

with col2:
    if st.session_state.opencv_image is not None:
        chunk_size = st.session_state.chunk_size
        raw_image = color.rgb2gray(st.session_state.opencv_image)

        down_image = downscale_image(raw_image, chunk_size)

        cropped_i = 1 - down_image

        cropped = cropped_i[:cropped_i.shape[0] // chunk_size * chunk_size, :cropped_i.shape[1] // chunk_size * chunk_size]
        
        coherence, two_phi = coh_ang_calc(cropped)

        all_img = orient_hsv(cropped, coherence, two_phi, mode='all')
        coh_img = orient_hsv(cropped, coherence, two_phi, mode='coherence')
        ang_img = orient_hsv(cropped, coherence, two_phi, mode='angle')
        # orient_hsv(cropped, structure_tensor_list, chunk_size=5, mode='angle')





        st.image([down_image, all_img, coh_img, ang_img])
    else:
        st.write("No image uploaded yet")