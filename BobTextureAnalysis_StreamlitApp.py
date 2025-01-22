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

    # Everything in this block will wait until submitted - it should contain uploading the images and infrequently changed parameters - initial downscale, etc.
    with st.form("form1", enter_to_submit=False, clear_on_submit=False):
        msg = "Upload a 2D image to be analyzed. Downsizing is reccomended for larger images"


        uploaded_image = st.file_uploader(msg, type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files=False)

        # TODO: take this out of the form, because inverting the images doesn't (shouldn't) change the coherence or angle calculation, but changing
        # if inverted forces recalculation
        # Change to parameter for orient_hsv?

        # No idea what label will make sense for this
        rescale_factor = st.number_input("Downscale factor (1 for no downscale - 0.01 for 100x smaller)", min_value=0.01, max_value=1.0, step=0.01, value=1.0)

        # Image
        if "opencv_image" not in st.session_state:
            st.session_state.raw_image_gray = None

        # Reading image if it has been uploaded
        if uploaded_image is not None:
            image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            cv_image = cv.imdecode(image_bytes, 1)
            cv_image_gray = color.rgb2gray(cv_image) # Convert to grayscale
            
            # rescale with skimage
            cv_image_rescaled = rescale(cv_image_gray, rescale_factor, anti_aliasing=True)

            st.session_state.raw_image_gray = cv_image_rescaled
        
        submit_button = st.form_submit_button("Analyze image")
    
    col1a, col1b = st.columns(2) # col1a is for displaying image, 1b is for parameters
    with col1a:
        st.write("Input image (grayscale)")
        if st.session_state.raw_image_gray is not None:
            st.image(st.session_state.raw_image_gray, use_container_width=True)
    
    with col1b:
        st.write("Processing options")

        # These don't need checking for NaN, because they have default values
        st.session_state.inner_sigma = st.number_input(value=1, min_value=1, max_value=100, step=1, label="sigma value (1 to 100 pixels)",
        help="Smaller values will emphasize higher frequecy detail, while larger values will focus on larger detail. Find what looks cool!")

        st.session_state.angle_phase_shift = st.number_input("Angle phase shift (0 to 180 degrees)", min_value=0, max_value=180, step=1)

        st.session_state.epsilon = st.number_input(min_value=1e-8, max_value=1.0, value=1e-8, label="epsilon (increasing can reduce coherence instability for near-constant reigons)")

        st.session_state.invert = st.checkbox(label="Invert image?", value=False)

        st.session_state.coh_ang_dark = st.checkbox(label="Dark mode for coh&ang ", value=False)
# col2 should be for visualizing processed images, and should have everything update live.
# Add dropdown menu for which layers to view: intensity, angle, and coherence - done
with col2:
    # Selection for which image to view
    imageToDisplay = st.selectbox("Image to display:", ("Intensity, Coherence, and Angle", "Coherence and Angle only", "Coherence only", "Angle only (black & white)"))
    if st.session_state.raw_image_gray is not None:
        raw_image_gray = st.session_state.raw_image_gray

        # calculate coherence and angle at a given sigma inner scale
        coherence, two_phi = coh_ang_calc(raw_image_gray, sigma_inner=st.session_state.inner_sigma, epsilon=st.session_state.epsilon)
        two_phi *= -1 # flip direction of increasing angles to CCW
        
        # This feels inefficient
        if "coh_ang" not in st.session_state:
            st.session_state.coh_ang = None
        st.session_state.coh_ang = (coherence, two_phi)

        all_img = orient_hsv(raw_image_gray, coherence, two_phi, mode='all', angle_phase=st.session_state.angle_phase_shift, invert = st.session_state.invert)
        coh_img = orient_hsv(raw_image_gray, coherence, two_phi, mode='coherence')
        ang_img = orient_hsv(raw_image_gray, coherence, two_phi, mode='angle', angle_phase=st.session_state.angle_phase_shift, night_mode=st.session_state.coh_ang_dark)
        ang_img_bw = orient_hsv(raw_image_gray, coherence, two_phi, mode="angle_bw", angle_phase=st.session_state.angle_phase_shift)
    
    # Display image based on user selection
        if imageToDisplay == "Intensity, Coherence, and Angle":
            image_to_show = all_img
        elif imageToDisplay == "Coherence and Angle only":
            image_to_show = ang_img
        elif imageToDisplay == "Coherence only":
            image_to_show = coh_img
        else:
            image_to_show = ang_img_bw
        st.image(image_to_show, use_container_width=True)

        col2a, col2b = st.columns(2)
        with col2a:
            k = st.number_input("choose k for (k by k) block reduce", min_value=1, max_value=100, value=1, step=1)
            (h, w) = raw_image_gray.shape
            raw_image_gray_small = raw_image_gray[:h//k * k, :w//k * k].reshape(h//k, k, w//k, k).mean(axis=(1, 3))
            coherence_small, two_phi_small = downscale_coh_ang(coherence, two_phi, k)
            all_img_small = orient_hsv(raw_image_gray_small, coherence_small, two_phi_small, mode='all', angle_phase=st.session_state.angle_phase_shift, invert = st.session_state.invert)
            coh_img_small = orient_hsv(raw_image_gray_small, coherence_small, two_phi_small, mode='coherence')
            ang_img_small = orient_hsv(raw_image_gray_small, coherence_small, two_phi_small, mode='angle', angle_phase=st.session_state.angle_phase_shift, night_mode=st.session_state.coh_ang_dark)
            ang_img_bw_small = orient_hsv(raw_image_gray_small, coherence_small, two_phi_small, mode="angle_bw", angle_phase=st.session_state.angle_phase_shift)
        with col2b:
            if imageToDisplay == "Intensity, Coherence, and Angle":
                image_to_show2 = all_img_small
            elif imageToDisplay == "Coherence and Angle only":
                image_to_show2 = ang_img_small
            elif imageToDisplay == "Coherence only":
                image_to_show2 = coh_img_small
            else:
                image_to_show2 = ang_img_bw_small
            st.image(image_to_show2, use_container_width=False)
    else:
        st.write("No image uploaded yet - click \"Analyze\"?")
    
    
    