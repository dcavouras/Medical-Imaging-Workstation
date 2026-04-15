import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal

# --- INTEGRATED FUNCTIONS FROM YOUR MODULES ---

def imNormalize(w, tones=256):
    mx, mn = np.max(w), np.min(w)
    if mx == mn: return w
    return np.round((tones - 1) * (w - mn) / (mx - mn))

def design2dFilter(im, fh):
    """Exact logic from your moduleFreqEnhancementFunctions.py"""
    y, x = np.shape(im)
    FH = np.zeros(np.shape(im), dtype=float)
    for k in range(y):
        for m in range(x):
            K = y/2 - k + 1
            M = x/2 - m + 1
            ir = int(np.sqrt((K*K + M*M)) + 0.5)
            if ir >= len(fh):
                ir = len(fh) - 1
            FH[k][m] = fh[ir]
    return np.fft.fftshift(FH)

def simpleWindow(im, wc, ww, image_depth, tones):
    """Exact windowing logic from your module"""
    im1 = np.asarray(im, dtype=float)
    Vb = (2.0 * wc + ww) / 2.0
    Va = Vb - ww
    if Vb > image_depth: Vb = image_depth
    if Va < 0: Va = 0
    
    M, N = np.shape(im1)
    for i in range(M):
        for j in range(N):
            if (im1[i][j] >= Va) and (im1[i][j] <= Vb):
                im1[i][j] = ((tones - 1) * (im1[i][j] - Va) / (Vb - Va))
            elif im1[i][j] < Va:
                im1[i][j] = 0
            elif im1[i][j] > Vb:
                im1[i][j] = tones - 1
    return im1

def GaussianMTF(N):
    fh = np.zeros(int(N), dtype=float)
    L = int(N // 2 + 1) if (N % 2 == 0) else int(N // 2 + 0.5)
    sigma = L/3 - 1
    for k in range(int(L)):
        fh[k] = np.exp(-k**2 / (2 * sigma**2))
    for k in range(int(N // 2 + 1), int(N)):
        fh[k] = fh[int(N - k)]
    return fh

# --- 1D FILTER WRAPPERS (FROM module_FreqDom_1d_Filters.py) ---
# Importing directly from your uploaded module functions logic

import module_FreqDom_1d_Filters as Filt

# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="BME Chapter 3 Full")
def run_chapter_3(shared_img, file_name):
    st.title("🌐 Chapter 3: Frequency Domain Filtering Suite")

    with st.sidebar:
        st.header("1. Data Input")
        uploaded_file = st.file_uploader("Upload Medical Image", type=['bmp', 'jpg', 'png'])
        if shared_img is not None: im_real = shared_img.astype(float)
        
        st.header("2. Filter Category")
        # All choices from your 'Choose_filter_type' variable
        choice = st.selectbox("Select Filter Class", 
                            ["Ideal", "Butterworth", "Exponential", "Gaussian", "Wiener (Deconvolution)"])
        
        # Context-aware settings
        if choice != "Wiener (Deconvolution)":
            mode = st.radio("Mode", ["LP (Low Pass)", "HP (High Pass)", "BR (Band Reject)", "BP (Band Pass)"])
            mode_idx = {"LP (Low Pass)":1, "HP (High Pass)":2, "BR (Band Reject)":3, "BP (Band Pass)":4}[mode]
            
            fco_default = 80 # default fco
            if choice == "Ideal": fco_default = 100
            
            fco = st.slider("Cut-off Frequency (fco)", 1, 300, fco_default)
            ndegree = st.slider("Order/Degree (n)", 1, 5, 2)
            trans = st.slider("Transition/Shift", 0, 200, 60)
            bw = st.slider("Band Width (w) - for BR/BP only", 1, 50, 15)
        else:
            wiener_mode = st.radio("Wiener Variant", ["Inverse Filter", "Wiener Filter", "Power Filter"])
            wiener_idx = {"Inverse Filter":1, "Wiener Filter":2, "Power Filter":3}[wiener_mode]

    if uploaded_file:
        # Load and setup
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        im = imNormalize(img_raw.astype(float), 256)
        M, N = im.shape
        Flength = int(np.round(np.sqrt(M*M + N*N)))

        # BRANCHING LOGIC MATCHING YOUR MAIN SCRIPT
        if choice == "Ideal":
            # Using 0.4 enh for HP as per your script's line 23
            enh = 0.4 if (mode_idx == 2 or mode_idx == 4)  else 0.0
            fh, sText = Filt.Ideal(Flength, fco, mode_idx, enh, trans, bw)
            
        elif choice == "Butterworth":
            fh, sText = Filt.Butterworth(Flength, ndegree, fco, mode_idx, trans)
            
        elif choice == "Exponential":
            fh, sText = Filt.Exponential(Flength, ndegree, fco, mode_idx, trans)
            
        elif choice == "Gaussian":
            fh, sText = Filt.Gaussian(Flength, ndegree, fco, mode_idx, trans)
            
        elif choice == "Wiener (Deconvolution)":
            # Logic from your generalizedWienerFilter function
            SIGMA = np.std(im)
            C = 20/SIGMA
            a, b = {"Inverse Filter":(0.2, 0.9), "Wiener Filter":(0, 1), "Power Filter":(0.5, 1)}[wiener_mode]
            sText = wiener_mode
            
            mtf = GaussianMTF(Flength)
            fh = np.zeros(int(Flength), dtype=float)
            for k in range(int(Flength)):
                fh[k] = (mtf[k]**2 / (mtf[k]**2 + b*C))**(1-a) / (mtf[k] + 0.0001)
                if fh[k] < C:
                    fh[k] = (mtf[k]**2 / (mtf[k]**2 + b*C))**(1-a) / C
            fh = fh / np.max(fh)

        # PROCESS: FFT -> Filter -> IFFT
        FH = design2dFilter(im, fh)
        Fim = np.fft.fft2(im)
        Fim_filtered = Fim * FH
        im_out = np.real(np.fft.ifft2(Fim_filtered))
        
        # Post-Processing
        im_out = imNormalize(im_out, 256)
        wc = st.sidebar.slider("Window Center (wc)", 0, 255, 130)
        ww = st.sidebar.slider("Window Width (ww)", 1, 255, 200)
        im_final = simpleWindow(im_out, wc, ww, 255, 256)

        # --- DISPLAY (Matching your 4-panel lecture style) ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(img_raw, use_container_width=True)
            
            # 1D Graph
            fig1, ax1 = plt.subplots(figsize=(6,4))
            ax1.plot(fh, label='filter')
            ax1.plot([Flength/2, Flength/2], [0, 1], 'b--', lw=3, label='axis of symmetry')
            ax1.plot(np.fft.fftshift(fh), 'r--', label='filter shifted to N/2')
            ax1.set_title(f"1-d filter: {sText}")
            ax1.set_xlabel('spatial frequencies'); ax1.set_ylabel('amplitude')
            ax1.legend(); ax1.grid()
            st.pyplot(fig1)

        with col2:
            st.subheader(f"Processed: {sText}")
            st.image(im_final.astype(np.uint8), use_container_width=True)
            
            # 2D Mask
            st.subheader("2-d Filter Mask (Shifted)")
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ampl = 255 * np.real(np.fft.fftshift(FH))
            ax2.imshow(ampl, cmap='gray', vmin=0, vmax=255)
            ax2.axis('off')
            st.pyplot(fig2)

        # Download
        is_success, buffer = cv2.imencode(".png", im_final.astype(np.uint8))
        st.sidebar.download_button("💾 Download Result", buffer.tobytes(), "chapter3_result.png")

    else:
        st.info("Upload a medical image (e.g., AA1a.bmp) to explore the Frequency Domain.")