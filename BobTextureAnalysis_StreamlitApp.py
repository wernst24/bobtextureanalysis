import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import time
import sys
# from functools import partial
# import pandas as pd

sys.dont_write_bytecode = True
sys.tracebacklimit = 0
from modules import *

st.set_page_config(
    page_title="BobTextureAnalysis",
    page_icon="🔬",
    layout="wide",
)

col1, col2 = st.columns(2)

# col1 should be for uploading image only: upload image, choose downscaling factor, and then preview at bottom.
with col1:
    # title for form
    st.markdown("# BobTextureAnalysis")
    with st.form("form1", enter_to_submit=False, clear_on_submit=False):
        msg = "Upload a 2D image to be analyzed. Downsizing is reccomended for larger images"

        uploaded_file = st.file_uploader(msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False)

        invert = st.checkbox(label="Invert image?", value=False)

        rescale_factor = st.number_input("Downscale percentage (1 for no downscale - 0.01 for 100x smaller)", min_value=0.01, max_value=1.0, step=0.01, value=1.0)

        # Image
        if "opencv_image" not in st.session_state:
            st.session_state.opencv_image = None

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            cv_image = cv.imdecode(file_bytes, 1)
            cv_image_gray = color.rgb2gray(cv_image)
            # cv_image_gray = cv_image
            if invert:
                cv_image_gray = 1 - cv_image_gray
            cv_image_rescaled = rescale(cv_image_gray, rescale_factor, anti_aliasing=True)

            st.session_state.opencv_image = cv_image_rescaled
        
        submit_button = st.form_submit_button("Analyze image")
    

    col1a, col1b = st.columns(2)
    with col1a:
        st.write("Input image (grayscale)")
        if st.session_state.opencv_image is not None:
            st.image(st.session_state.opencv_image, use_container_width=True)
    
    with col1b:
        st.write("Processing options")
        st.session_state.inner_sigma = st.number_input(value=1, min_value=1, max_value=100, step=1, label="sigma value (1 to 100 pixels)",
        help="Smaller values will emphasize higher frequecy detail, while larger values will focus on larger detail. Find what looks cool!")
        # if "inner_sigma" not in st.session_state:
        #     st.session_state.inner_sigma = None

        # if inner_sigma: 
        #      = inner_sigma

        st.session_state.angle_phase_shift = st.number_input("Angle phase shift (0 to 180 degrees)", min_value=0, max_value=180, step=1)

        # if "angle_phase_shift" not in st.session_state:
        #         st.session_state.angle_phase_shift = None

        

        st.session_state.epsilon = st.number_input(min_value=1e-8, max_value=1.0, value=1e-8, label="epsilon (increasing can reduce coherence instability for near-constant reigons)")
        # if "epsilon" not in st.session_state:
        #     st.session_state.epsilon = None
        #  = epsilon

# col2 should be for visualizing processed images, and should have everything update live.
# Add dropdown menu for which layers to view: intensity, angle, and coherence
with col2:

    imageToDisplay = st.selectbox("Image to display:", ("Intensity, Coherence, and Angle", "Coherence and Angle only", "Coherence only", "Angle only (black & white)"))
    if st.session_state.opencv_image is not None:
        raw_image_gray = st.session_state.opencv_image

        coherence, two_phi = coh_ang_calc(raw_image_gray, sigma_inner=st.session_state.inner_sigma, gradient_mode='sobel', epsilon=st.session_state.epsilon)
        two_phi *= -1
        if "coh_ang" not in st.session_state:
            st.session_state.coh_ang = None
        st.session_state.coh_ang = (coherence, two_phi)

        all_img = orient_hsv(raw_image_gray, coherence, two_phi, mode='all', angle_phase=st.session_state.angle_phase_shift)
        coh_img = orient_hsv(raw_image_gray, coherence, two_phi, mode='coherence')
        ang_img = orient_hsv(raw_image_gray, coherence, two_phi, mode='angle', angle_phase=st.session_state.angle_phase_shift)
        ang_img_bw = orient_hsv(raw_image_gray, coherence, two_phi, mode="angle_bw", angle_phase=st.session_state.angle_phase_shift)
    
    if st.session_state.opencv_image is not None:
        if imageToDisplay == "Intensity, Coherence, and Angle":
            image_to_show = all_img
        elif imageToDisplay == "Coherence and Angle only":
            image_to_show = ang_img
        elif imageToDisplay == "Coherence only":
            image_to_show = coh_img
        else:
            image_to_show = ang_img_bw
        st.image(image_to_show, use_container_width=True)
    else:
        st.write("No image uploaded yet - click \"Analyze\"?")