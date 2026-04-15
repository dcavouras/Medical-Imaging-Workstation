import streamlit as st
import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import os  # Added to handle filenames

# --- CORE PROCESSING FUNCTIONS ---
def imNormalize(w, tones=256):
    mx, mn = np.max(w), np.min(w)
    return np.round((tones - 1) * (w - mn) / (mx - mn)) if mx != mn else w

def simpleWindow(im, wc, ww, image_depth=255, tones=256):
    Vb = (2.0 * wc + ww) / 2.0
    Va = Vb - ww
    im1 = np.clip(im, max(0, Va), min(image_depth, Vb))
    return ((tones - 1) * (im1 - Va) / (ww))

# --- KERNEL DATABASE ---
KERNELS = {
    "Smoothing": [
        [[0,1,0],[1,1,1],[0,1,0]], [[1,1,1],[1,1,1],[1,1,1]], 
        [[1,1,1],[1,2,1],[1,1,1]], [[1,2,1],[2,4,2],[1,2,1]]
    ],
    "Edge Enhancement": [
        [[0,1,0],[1,-4,1],[0,1,0]], [[1,1,1],[1,-8,1],[1,1,1]],
        [[1,2,1],[2,-12,2],[1,2,1]], [[-1,2,-1],[2,-4,2],[-1,2,-1]]
    ],
    "High Emphasis": [
        [[0,-1,0],[-1,5,-1],[0,-1,0]], [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]],
        [[-1,-2,-1],[-2,13,-2],[-1,-2,-1]], [[1,-2,1],[-2,5,-2],[1,-2,1]]
    ]
}

st.set_page_config(layout="wide", page_title="BME Image Processing")
def run_chapter_2(shared_img, file_name):
    st.title("🔬 Chapter 2: Spatial Domain Processing")

    with st.sidebar:
        st.header("1. Upload")
        uploaded_file = st.file_uploader("Upload Biomedical Image", type=['bmp', 'jpg', 'png'])
        if shared_img is not None: im_real = shared_img.astype(float)
        op_type = st.radio("Method", ["Kernel Convolution", "Median Filter"])


    if uploaded_file:
        # --- FILENAME HANDLING ---
        # Split filename and extension: "AA1a" and ".bmp"
        filename_core, file_extension = os.path.splitext(uploaded_file.name)
        output_filename = f"{filename_core}_processed{file_extension}"
        
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        im_float = img.astype(float)

        if op_type == "Kernel Convolution":
            category = st.selectbox("Select Category", list(KERNELS.keys()))
            st.write(f"**Select a {category} Mask:**")
            cols = st.columns(4)
            for i in range(4):
                with cols[i]:
                    st.code(np.array(KERNELS[category][i]))
                    if st.button(f"Use Mask {i+1}", key=f"btn_{i}"):
                        st.session_state['mask_choice'] = i
            
            idx = st.session_state.get('mask_choice', 0)
            kernel = np.array(KERNELS[category][idx], dtype=float)
            sK = np.sum(kernel)
            if sK > 0: kernel /= sK
            im_out = signal.convolve2d(im_float, kernel, mode='same')
            im_out = np.clip(im_out, 0, 255)
            st.success(f"Applying Mask {idx+1}")

        else:
            im_out = signal.medfilt2d(im_float, (3,3))
            wc = st.select_slider("Window Center (Level)", options=range(256), value=130)
            ww = st.select_slider("Window Width", options=range(1, 256), value=170)
            im_out = simpleWindow(im_out, wc, ww)

        im_final = imNormalize(im_out).astype(np.uint8)

        # --- DISPLAY ---
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Input Image")
            st.image(img, use_container_width=True)
            fig_in, ax_in = plt.subplots(figsize=(5,2))
            ax_in.hist(img.ravel(), bins=256, color='gray')
            st.pyplot(fig_in)
        with c2:
            st.subheader("Processed Result")
            st.image(im_final, use_container_width=True)
            fig_out, ax_out = plt.subplots(figsize=(5,2))
            ax_out.hist(im_final.ravel(), bins=256, color='blue')
            st.pyplot(fig_out)

        # --- EXPORT WITH DYNAMIC FILENAME & TYPE ---
        with st.sidebar:
            st.header("3. Export")
            # Use the same extension for encoding (e.g., ".bmp" or ".jpg")
            is_success, buffer = cv2.imencode(file_extension, im_final)
            if is_success:
                st.download_button(
                    label=f"💾 Download {output_filename}",
                    data=buffer.tobytes(),
                    file_name=output_filename,
                    mime=f"image/{file_extension[1:].replace('jpg', 'jpeg')}" # simple mime type fix
                )

    else:
        st.info("Please upload an image to begin the lab.")