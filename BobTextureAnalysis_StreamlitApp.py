import streamlit as st

import numpy as np

import matplotlib.pyplot as plt

import time

from io import BytesIO

import sys

sys.dont_write_bytecode = True

sys.tracebacklimit = 0

from modules import *

with st.form(key='form1', clear_on_submit=False):
    st.markdown('test')

    uploaded_file = st.file_uploader("Upload a 2D image to be analyzed. Works best with images smaller than 600x600 pixels.", type=["tif", "tiff", "png", "jpg", "jpeg"], accept_multiple_files = False, label_visibility = 'collapsed')

    st.form_submit_button()