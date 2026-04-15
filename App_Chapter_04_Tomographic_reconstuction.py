import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import radon, iradon, iradon_sart
from PIL import Image
import io

# Helper functions from your teaching script
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

st.set_page_config(layout="wide")
def run_chapter_4(shared_img,file_name):
    st.title("Chapter 4: Tomographic Image Reconstruction")
    st.write("Educational tool for BME students - Flexible File Handling")

    # --- SIDEBAR: PARAMETER INPUTS ---
    st.sidebar.header("1. Data Input")

    # File Uploader: This allows you to browse your computer for an image
    uploaded_file = st.sidebar.file_uploader("Upload Patient Slice (BMP, PNG, JPG)", type=["bmp", "png", "jpg"])

    st.sidebar.header("2. Save Settings")
    # Text Input for Save Directory: Allows you to specify where to save
    save_dir = st.sidebar.text_input("Destination Directory (Absolute path or relative)", value="./results")

    st.sidebar.header("3. Scan Parameters")
    n_proj = st.sidebar.slider("Total degrees of camera revolution", 10, 360, 180)
    tomo_algorithm = st.sidebar.radio("Tomographic Algorithm", ["Filtered Back Projection (FBP)", "SART (ART)"])

    if tomo_algorithm == "Filtered Back Projection (FBP)":
        filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', 'None']
        filter_choice = st.sidebar.selectbox("Reconstruction Filter", filters)
    else:
        iterations = st.sidebar.number_input("Number of iterations", min_value=1, max_value=20, value=5)

    # --- EXECUTION ---
    if st.sidebar.button("Run Reconstruction") and uploaded_file is not None:
        # Read the uploaded file
        im_raw = Image.open(uploaded_file)
        im = np.array(im_raw)
        
        # Pre-processing logic from your script
        if len(im.shape) == 3:
            im = rgb2gray(im)
        im = np.asarray(im, float)
        
        # Normalization
        max_A, min_A = np.max(im), np.min(im)
        A = (im - min_A) * (255 / (max_A - min_A))
        
        # 1. Radon Transform (Sinogram)
        theta = np.arange(0, n_proj)
        sinogram = radon(A, theta=theta, circle=False)
        
        # 2. Reconstruction path selection
        if tomo_algorithm == "Filtered Back Projection (FBP)":
            f_name = None if filter_choice == "None" else filter_choice
            reconstructed = iradon(sinogram, theta=theta, filter_name=f_name, output_size=len(A))
        else:
            I_ART = iradon_sart(sinogram, theta=theta)
            for i in range(iterations - 1):
                I_ART = iradon_sart(sinogram, theta=theta, image=I_ART)
            
            # Slicing logic to match original size
            M, N = I_ART.shape
            m, n = A.shape
            reconstructed = I_ART[int(M/2)-int(m/2):int(M/2)+int(m/2), int(N/2)-int(n/2):int(N/2)+int(n/2)]

        # --- SAVING LOGIC ---
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_filename = f"reconstructed_{uploaded_file.name}"
        save_path = os.path.join(save_dir, save_filename)
        
        # Convert back to uint8 for standard image saving
        recon_to_save = np.clip(reconstructed, 0, 255).astype(np.uint8)
        Image.fromarray(recon_to_save).save(save_path)
        
        st.success(f"File saved to: {save_path}")

        # --- DISPLAY RESULTS ---
        # Updated to use 'use_container_width' to avoid depreciation errors
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original")
            st.image(im_raw, use_container_width=True)
        with col2:
            st.subheader("Sinogram")
            fig_sino, ax_sino = plt.subplots()
            ax_sino.imshow(sinogram, cmap='gray', aspect='auto')
            st.pyplot(fig_sino)
        with col3:
            st.subheader("Reconstructed")
            st.image(save_path, use_container_width=True)

    elif uploaded_file is None:
        st.info("Please browse and upload an image file from your computer.")