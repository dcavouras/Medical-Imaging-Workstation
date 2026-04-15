import streamlit as st
import numpy as np
import cv2
from PIL import Image

# --- CHAPTER IMPORTS ---
# We treat your existing apps as modules
import app_Chapter_01_image_display_manipulation_methods as Ch1
import app_Chapter_02_spatial_domain_image_processing as Ch2
import app_Chapter_03_frequency_domain_image_processing as Ch3
import App_Chapter_04_Tomographic_reconstuction as Ch4

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="BME Unified Imaging Suite")

# --- SHARED STATE MANAGEMENT ---
# This ensures an image uploaded in one chapter is available to all
if 'shared_image' not in st.session_state:
    st.session_state['shared_image'] = None
if 'file_name' not in st.session_state:
    st.session_state['file_name'] = ""

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063171.png", width=100)
st.sidebar.title("Workstation Menu")
app_mode = st.sidebar.selectbox("Navigate to Chapter:",
    ["Welcome Screen", 
     "Ch 1: Display & Histograms", 
     "Ch 2: Spatial Filtering", 
     "Ch 3: Frequency Domain", 
     "Ch 4: Tomographic Reconstruction"])

st.sidebar.markdown("---")

# --- SHARED FILE UPLOADER ---
# Instead of uploading in every chapter, we do it once here
# uploaded_file = st.sidebar.file_uploader("Upload Master Patient Image", type=["bmp", "png", "jpg", "jpeg", "dcm"])

# if uploaded_file:
#     # Process the file once and store in session_state
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
#     st.session_state['shared_image'] = img
#     st.session_state['file_name'] = uploaded_file.name
#     st.sidebar.success(f"Active Image: {uploaded_file.name}")

# --- APP ROUTING ---

if app_mode == "Welcome Screen":
    st.title("🏥 BME Medical Imaging Workstation")
    st.markdown("""
    ### Final Project Presentation Framework
    Welcome to the unified software suite. This tool integrates four core domains of medical image processing:
    1. **Display & Point Processing**: Histogram modification and windowing.
    2. **Spatial Domain**: Convolutional kernels and neighborhood operations.
    3. **Frequency Domain**: Fourier Transforms and spectral filtering.
    4. **Tomography**: Radon transform and image reconstruction from projections.
    
    **Instructions:** * Upload an image in the sidebar to begin.
    * Use the dropdown menu to move between processing stages.
    """)

elif app_mode == "Ch 1: Display & Histograms":
    # Pass the shared image into the Chapter 1 logic
    Ch1.run_chapter_1(st.session_state['shared_image'],st.session_state['file_name'])

elif app_mode == "Ch 2: Spatial Filtering":
    Ch2.run_chapter_2(st.session_state['shared_image'],st.session_state['file_name'])

elif app_mode == "Ch 3: Frequency Domain":
    Ch3.run_chapter_3(st.session_state['shared_image'],st.session_state['file_name'])

elif app_mode == "Ch 4: Tomographic Reconstruction":
    Ch4.run_chapter_4(st.session_state['shared_image'],st.session_state['file_name'])