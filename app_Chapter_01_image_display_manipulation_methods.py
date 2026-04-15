import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io

# DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

def run_chapter_1(shared_img,file_name):
    
    # ── PAGE TITLE ────────────────────────────────────────────────────────────────
    st.title("Lecture 2 - Display Modification Methods")
    st.subheader("Application on Real Medical Images")

    # ── SIDEBAR ───────────────────────────────────────────────────────────────────
    st.sidebar.title("Settings")
    method = st.sidebar.selectbox("Choose display method:",
                                ["Simple Display",
                                    "Simple Window",
                                    "Broken Window",
                                    "Double Window",
                                    "inverse",
                                    "logarithmic",
                                    "inverse logarithmic",
                                    "power",
                                    "sine-window",
                                    "exp-window",
                                    "Histogram Equalization",
                                    "CDF Equalization",
                                    "CLAHE"])

    image_depth = 256
    tones = 256

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameters")

    if method == "Simple Window":
        wc = st.sidebar.slider("Window Center (wc):", 0, 255, 100)
        ww = st.sidebar.slider("Window Width (ww):",  1, 255, 150)
    elif method == "Broken Window":
        gray_val = st.sidebar.slider("Gray Value:", 0, 255, 128)
        im_val   = st.sidebar.slider("Image Value:", 0, 255, 70)
    elif method == "Double Window":
        ww1 = st.sidebar.slider("Window Width 1 (ww1):", 1, 255, 100)
        wl1 = st.sidebar.slider("Window Level 1 (wl1):", 0, 255,  50)
        ww2 = st.sidebar.slider("Window Width 2 (ww2):", 1, 255, 100)
        wl2 = st.sidebar.slider("Window Level 2 (wl2):", 0, 255, 150)
    elif method == "logarithmic":
        r = st.sidebar.slider("r value:", 0.01, 0.2, 0.05)
    elif method == "inverse logarithmic":
        cValue = st.sidebar.slider("c value:", 50, 200, 128)
    elif method == "power":
        gamma = st.sidebar.slider("gamma:", 0.10, 0.90, 0.55)
    elif method == "sine-window":
        sValue = st.sidebar.slider("s value:", 1, 10, 5)
    elif method == "exp-window":
        eValue = st.sidebar.slider("e value:", 50, 200, 90)
    elif method == "CLAHE":
        clip_limit = st.sidebar.slider("Clip Limit:", 0.01, 0.10, 0.03)

    # ── IMAGE UPLOAD (sidebar) ────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Medical Image")
    uploaded_file = st.sidebar.file_uploader(
        "Upload image:",
        type=["bmp", "png", "jpg", "jpeg", "dcm"]
    )

    format_list = ["bmp", "png", "jpg", "jpeg"]
    save_format = "png"
    if uploaded_file is not None:
        original_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if original_ext == "jpeg": original_ext = "jpg"
        default_index = format_list.index(original_ext) if original_ext in format_list else 1
        save_format = st.sidebar.selectbox("Save format:", format_list, index=default_index)

    # ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
    def loadUploadedImage(uploaded_file):
        filename = uploaded_file.name.lower()
        if filename.endswith(".dcm"):
            if not DICOM_AVAILABLE:
                st.error("pydicom not installed. Run: pip install pydicom")
                return None
            dicom_data = pydicom.dcmread(uploaded_file)
            im = dicom_data.pixel_array.astype(float)
            if im.max() > 255:
                im = (im - im.min()) / (im.max() - im.min()) * 255.0
            return im
        else:
            image = Image.open(uploaded_file).convert("L")
            return np.asarray(image, dtype=float)

    def applySimpleWindow(im, wc, ww, image_depth, tones):
        Vb = (2.0*wc + ww) / 2.0
        if Vb > image_depth: Vb = image_depth
        Va = Vb - ww
        if Va < 0: Va = 0
        im1 = np.zeros(im.shape, dtype=float)
        N, M = im.shape
        for i in range(N):
            for j in range(M):
                Vm = im[i][j]
                if Vm < Va: t = 0
                elif Vm > Vb: t = tones - 1
                else: t = (tones-1) * (Vm-Va) / (Vb-Va)
                im1[i][j] = np.round(t)
        return im1

    def applyBrokenWindow(im, image_depth, tones, gray_val, im_val):
        im1 = np.zeros(im.shape, dtype=float)
        N, M = im.shape
        for i in range(N):
            for j in range(M):
                if im[i][j] <= im_val:
                    im1[i][j] = (gray_val / im_val) * im[i][j]
                else:
                    im1[i][j] = (((tones-1)-(gray_val+1)) / (image_depth-(im_val+1))) * (im[i][j]-(im_val+1)) + (gray_val+1)
        return np.round(im1)

    def applyDoubleWindow(im, ww1, wl1, ww2, wl2, image_depth, tones):
        im1 = np.zeros(im.shape, dtype=float)
        N, M = im.shape
        half = (tones/2) - 1
        ve1 = round((2.0*wl1+ww1)/2.0); vs1 = ve1 - ww1
        ve2 = round((2.0*wl2+ww2)/2.0); vs2 = ve2 - ww2
        if vs2 < ve1:
            new_point = round((vs2+ve1)/2.0)
            ve1 = new_point; vs2 = ve1
        if vs1 < 0: vs1 = 0
        if ve2 > image_depth: ve2 = image_depth
        for i in range(N):
            for j in range(M):
                if im[i][j] < vs1: im1[i][j] = 0
                if im[i][j] >= vs1 and im[i][j] <= ve1:
                    im1[i][j] = round(((half)/(ve1-vs1))*(im[i][j]-vs1))
                if im[i][j] > ve1 and im[i][j] < vs2:
                    im1[i][j] = half + 1
                if im[i][j] >= vs2 and im[i][j] <= ve2:
                    im1[i][j] = round((((tones-1)-(half+1))/(ve2-vs2))*(im[i][j]-vs2)+(half+1))
                if im[i][j] > ve2: im1[i][j] = tones - 1
        return im1

    def formNonLinearWindowFunction(tones, method):
        import math
        w = np.zeros(tones)
        for i in range(0, tones):
            if method == "inverse":
                w[i] = tones-i-1
            elif method == "logarithmic":
                w[i] = math.log(1+r*i)
            elif method == "inverse logarithmic":
                w[i] = np.exp(i)**(1/cValue)-1
            elif method == "power":
                w[i] = i**gamma
            elif method == "sine-window":
                w[i] = np.sin(2*np.pi*i/(sValue*(tones-1)))
            elif method == "exp-window":
                w[i] = 1-np.exp(-i/eValue)
        w = (tones-1)*((w-np.min(w))/(np.max(w)-np.min(w)))
        return w

    def modifyImageFromNonLinearFunction(im, w):
        N = np.size(im, 0); M = np.size(im, 1)
        mn = np.min(im); mx = np.max(im)
        im1 = np.round((tones-1)*(im-mn)/(mx-mn))
        for i in range(0, N):
            for j in range(M):
                v = int(im1[i, j])
                im1[i, j] = (w[v])
        return im1

    def f_histogram(A, image_depth, tones):
        B = A if np.max(A) <= (tones-1) else np.round((tones-1)*((A-0)/(image_depth-0)))
        M = np.size(B, 0); N = np.size(B, 1)
        Bval = np.reshape(B, M*N)
        h = np.zeros(tones, dtype=float)
        for i in range(np.size(Bval)):
            val = np.int16(Bval[i])
            h[val] = h[val]+1
        return h

    def f_hequalization(A, image_depth, tones):
        B = np.round((tones-1)*((A-0)/(image_depth-0)))
        M = np.size(B, 0); N = np.size(B, 1)
        Bval = np.reshape(B, M*N)
        p = np.argsort(Bval)
        neq = np.int32((M*N)/tones+0.5)
        BL = len(Bval)
        az = np.int32(np.fix((N*M)/neq))
        zRem = np.int32(np.remainder(BL, neq))
        D = np.zeros(M*N)
        k = -1
        for i in range(0, (neq*az), neq):
            k = k+1
            for j in range(0, neq):
                D[i+j] = k
        if zRem > 0:
            for i in range((neq*az), ((az*neq)+zRem)):
                D[i] = tones-1
        L = np.zeros(M*N)
        k = -1
        for i in range(M):
            for j in range(N):
                k = k+1; L[p[k]] = D[k]
        Z = np.reshape(L, B.shape)
        mx = np.max(Z); mn = np.min(Z)
        Z = np.round((tones-1)*(Z-mn)/(mx-mn))
        return Z

    def CDF_equalization(im, image_depth, tones):
        B = np.round((tones-1)*((im)/(np.max(im))))
        M = np.size(im, 0); N = np.size(im, 1)
        CDFh = np.zeros(tones, dtype=float)
        CDFq = np.zeros(tones, dtype=float)
        h = f_histogram(im, image_depth, tones)
        tone_values = ((M*N)/tones)
        q = (tone_values*np.ones(tones, dtype=float))
        for i in range(tones):
            for j in range(i+1):
                CDFh[i] = CDFh[i]+h[j]
                CDFq[i] = CDFq[i]+q[j]
        B = CDFh[np.int32(B)]/tone_values-1
        B = np.round(B)
        return B

    def histCumsum(im):
        hist, bins = np.histogram(im.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        return cdf_normalized

    def plotWindowingFunction(method, tones, image_depth, **kwargs):
        import math
        x = np.arange(0, image_depth)
        y = np.zeros(image_depth)
        if method == "Simple Display":
            y = (tones-1) / (image_depth-1) * x
        elif method == "Simple Window":
            wc = kwargs['wc']; ww = kwargs['ww']
            Vb = (2.0*wc + ww) / 2.0
            if Vb > image_depth: Vb = image_depth
            Va = Vb - ww
            if Va < 0: Va = 0
            for i, v in enumerate(x):
                if v < Va: y[i] = 0
                elif v > Vb: y[i] = tones - 1
                else: y[i] = (tones-1) * (v-Va) / (Vb-Va)
        elif method == "Broken Window":
            gray_val = kwargs['gray_val']; im_val = kwargs['im_val']
            for i, v in enumerate(x):
                if v <= im_val:
                    y[i] = (gray_val / im_val) * v
                else:
                    y[i] = (((tones-1)-(gray_val+1)) / (image_depth-(im_val+1))) * (v-(im_val+1)) + (gray_val+1)
        elif method == "Double Window":
            ww1=kwargs['ww1']; wl1=kwargs['wl1']
            ww2=kwargs['ww2']; wl2=kwargs['wl2']
            half = (tones/2) - 1
            ve1 = round((2.0*wl1+ww1)/2.0); vs1 = ve1 - ww1
            ve2 = round((2.0*wl2+ww2)/2.0); vs2 = ve2 - ww2
            if vs2 < ve1:
                new_point = round((vs2+ve1)/2.0)
                ve1 = new_point; vs2 = ve1
            if vs1 < 0: vs1 = 0
            if ve2 > image_depth: ve2 = image_depth
            for i, v in enumerate(x):
                if v < vs1: y[i] = 0
                elif v >= vs1 and v <= ve1: y[i] = ((half)/(ve1-vs1))*(v-vs1)
                elif v > ve1 and v < vs2: y[i] = half + 1
                elif v >= vs2 and v <= ve2: y[i] = (((tones-1)-(half+1))/(ve2-vs2))*(v-vs2)+(half+1)
                elif v > ve2: y[i] = tones - 1
        elif method == "inverse":
            for i in range(image_depth): y[i] = tones-i-1
        elif method == "logarithmic":
            for i in range(image_depth): y[i] = math.log(1+kwargs["r"]*i)
        elif method == "inverse logarithmic":
            for i in range(image_depth): y[i] = np.exp(i)**(1/kwargs["cValue"])-1
        elif method == "power":
            for i in range(image_depth): y[i] = i**kwargs["gamma"]
        elif method == "sine-window":
            for i in range(image_depth): y[i] = np.sin(2*np.pi*i/(kwargs["sValue"]*(tones-1)))
        elif method == "exp-window":
            for i in range(image_depth): y[i] = 1-np.exp(-i/kwargs["eValue"])
        y = (tones-1)*(y-min(y))/(max(y)-min(y))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(x, y, color='red', linewidth=2)
        ax.set_xlim(0, image_depth); ax.set_ylim(0, tones)
        ax.set_xlabel("Input gray value"); ax.set_ylabel("Output gray value")
        ax.set_title(method); ax.grid(True)
        plt.tight_layout()
        return fig

    def plotHistogramFunction(im, im1, method):
        """For histogram methods: show cumulative histograms instead of display function."""
        y_init = histCumsum(im)
        w      = histCumsum(im1)
        x      = np.arange(0, len(y_init))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(x, y_init, color='red',   linewidth=2, label='Original')
        ax.plot(x, w,      color='green', linewidth=2, label='Processed')
        ax.set_xlabel("Gray value"); ax.set_ylabel("Cumulative count")
        ax.set_title("Cumulative Histogram"); ax.legend(); ax.grid(True)
        plt.tight_layout()
        return fig

    def prepareDownload(im_out, save_format):
        im_pil = Image.fromarray(np.clip(im_out, 0, 255).astype(np.uint8))
        buf = io.BytesIO()
        if save_format in ["jpg", "jpeg"]:
            im_pil.save(buf, format="JPEG", quality=95); ext = "jpg"
        elif save_format == "bmp":
            im_pil.save(buf, format="BMP"); ext = "bmp"
        else:
            im_pil.save(buf, format="PNG"); ext = "png"
        buf.seek(0)
        return buf, ext

    # ── MAIN LOGIC ────────────────────────────────────────────────────────────────
    HISTOGRAM_METHODS = ["Histogram Equalization", "CDF Equalization", "CLAHE"]

    if uploaded_file is not None:

        im_real = loadUploadedImage(uploaded_file)

        if im_real is not None:
            im_real = 255.0 * im_real / np.max(im_real)

            # ── Apply selected method ─────────────────────────────────────────────
            if method == "Simple Display":
                mn = np.min(im_real); mx = np.max(im_real)
                im_out = np.round((tones-1) * (im_real-mn) / (mx-mn))
                fig = plotWindowingFunction(method, tones, image_depth)
            elif method == "Simple Window":
                im_out = applySimpleWindow(im_real, wc, ww, image_depth, tones)
                fig = plotWindowingFunction(method, tones, image_depth, wc=wc, ww=ww)
            elif method == "Broken Window":
                im_out = applyBrokenWindow(im_real, image_depth, tones, gray_val, im_val)
                fig = plotWindowingFunction(method, tones, image_depth, gray_val=gray_val, im_val=im_val)
            elif method == "Double Window":
                im_out = applyDoubleWindow(im_real, ww1, wl1, ww2, wl2, image_depth, tones)
                fig = plotWindowingFunction(method, tones, image_depth, ww1=ww1, wl1=wl1, ww2=ww2, wl2=wl2)
            elif method == "inverse":
                w = formNonLinearWindowFunction(tones, method)
                im_out = modifyImageFromNonLinearFunction(im_real, w)
                fig = plotWindowingFunction(method, tones, image_depth)
            elif method == "logarithmic":
                w = formNonLinearWindowFunction(tones, method)
                im_out = modifyImageFromNonLinearFunction(im_real, w)
                fig = plotWindowingFunction(method, tones, image_depth, r=r)
            elif method == "inverse logarithmic":
                w = formNonLinearWindowFunction(tones, method)
                im_out = modifyImageFromNonLinearFunction(im_real, w)
                fig = plotWindowingFunction(method, tones, image_depth, cValue=cValue)
            elif method == "power":
                w = formNonLinearWindowFunction(tones, method)
                im_out = modifyImageFromNonLinearFunction(im_real, w)
                fig = plotWindowingFunction(method, tones, image_depth, gamma=gamma)
            elif method == "sine-window":
                w = formNonLinearWindowFunction(tones, method)
                im_out = modifyImageFromNonLinearFunction(im_real, w)
                fig = plotWindowingFunction(method, tones, image_depth, sValue=sValue)
            elif method == "exp-window":
                w = formNonLinearWindowFunction(tones, method)
                im_out = modifyImageFromNonLinearFunction(im_real, w)
                fig = plotWindowingFunction(method, tones, image_depth, eValue=eValue)
            elif method == "Histogram Equalization":
                im_out = f_hequalization(im_real, image_depth, tones)
                fig = plotHistogramFunction(im_real, im_out, method)
            elif method == "CDF Equalization":
                im_out = CDF_equalization(im_real, image_depth, tones)
                fig = plotHistogramFunction(im_real, im_out, method)
            elif method == "CLAHE":
                from skimage import exposure
                im_out = exposure.equalize_adapthist(np.uint8(im_real), clip_limit=clip_limit)
                im_out = 255.0 * im_out / np.max(im_out)
                fig = plotHistogramFunction(im_real, im_out, method)

            # ── Display ───────────────────────────────────────────────────────────
            im_out_display  = np.clip(im_out,  0, 255).astype(np.uint8)
            im_real_display = np.clip(im_real, 0, 255).astype(np.uint8)

            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.subheader("Original Image")
                st.image(im_real_display, use_container_width=True)
                st.caption(uploaded_file.name)

            with col2:
                if method in HISTOGRAM_METHODS:
                    st.subheader("Cumulative Histogram")
                else:
                    st.subheader("Display Function")
                st.pyplot(fig)

            with col3:
                st.subheader("Processed: " + method)
                st.image(im_out_display, use_container_width=True)

            # ── Download ──────────────────────────────────────────────────────────
            st.markdown("---")
            buf, ext = prepareDownload(im_out, save_format)
            original_name = uploaded_file.name.rsplit(".", 1)[0]
            download_name = f"processed_{original_name}.{ext}"
            st.download_button(
                label=f"💾 Download Processed Image (.{ext})",
                data=buf, file_name=download_name,
                mime=f"image/{ext}"
            )
            st.caption(f"Will be saved as: {download_name}")

    else:
        st.info("👆 Please upload a medical image using the sidebar  (BMP, PNG, JPG, JPEG or DCM)")
        if not DICOM_AVAILABLE:
            st.warning("⚠️ DICOM support not available. Install with: pip install pydicom")
