import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import time

st.set_page_config(page_title="Contactless Heart Rate Monitor", layout="wide")
st.title("💓 Contactless Heart Rate Monitor")
st.caption("Remote photoplethysmography (rPPG) — estimating heart rate from facial video using a standard webcam.")

mp_face_detection = mp.solutions.face_detection

def bandpass_filter(data, lowcut=0.7, highcut=3.0, fs=30, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_status(bpm):
    if bpm < 60:
        return "Low", "blue"
    elif bpm <= 100:
        return "Normal / Resting", "green"
    else:
        return "Elevated", "red"

col1, col2 = st.columns([2, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    bpm_placeholder = st.empty()
    status_placeholder = st.empty()
    chart_placeholder = st.empty()

start = st.checkbox("Start Camera")
reset = st.button("Reset Measurement")

if "signal_buffer" not in st.session_state or reset:
    st.session_state.signal_buffer = []
    st.session_state.time_buffer = []
    st.session_state.bpm_history = []
    st.session_state.bpm_locked = False
    st.session_state.locked_bpm = None

BUFFER_SECONDS = 15
STABILITY_THRESHOLD = 3

if start:
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
    cap = cv2.VideoCapture(0)

    while start:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        filtered_for_plot = None

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            box_w = int(bboxC.width * w)
            box_h = int(bboxC.height * h)

            forehead_x = x + int(box_w * 0.3)
            forehead_y = y + int(box_h * 0.02)
            forehead_w = int(box_w * 0.4)
            forehead_h = int(box_h * 0.15)

            cv2.rectangle(frame, (forehead_x, forehead_y),
                          (forehead_x + forehead_w, forehead_y + forehead_h),
                          (255, 0, 0), 2) 

            roi = frame[forehead_y:forehead_y + forehead_h, forehead_x:forehead_x + forehead_w]

            if not st.session_state.bpm_locked and roi.size > 0:
                green_avg = np.mean(roi[:, :, 1])
                st.session_state.signal_buffer.append(green_avg)
                st.session_state.time_buffer.append(time.time())

            tb = st.session_state.time_buffer
            sb = st.session_state.signal_buffer
            while tb and (tb[-1] - tb[0]) > BUFFER_SECONDS:
                sb.pop(0)
                tb.pop(0)

            if not st.session_state.bpm_locked and len(sb) > 90:
                try:
                    fs = len(sb) / (tb[-1] - tb[0])
                    filtered = bandpass_filter(np.array(sb), fs=fs)
                    filtered_for_plot = filtered

                    freqs = fftfreq(len(filtered), d=1/fs)
                    fft_vals = np.abs(fft(filtered))

                    valid = (freqs > 0.7) & (freqs < 3.0)
                    if np.any(valid):
                        peak_freq = freqs[valid][np.argmax(fft_vals[valid])]
                        bpm = peak_freq * 60

                        st.session_state.bpm_history.append(bpm)
                        if len(st.session_state.bpm_history) > 5:
                            st.session_state.bpm_history.pop(0)

                        bh = st.session_state.bpm_history
                        smoothed_bpm = np.mean(bh)

                        if len(bh) == 5 and (max(bh) - min(bh)) < STABILITY_THRESHOLD:
                            st.session_state.bpm_locked = True
                            st.session_state.locked_bpm = smoothed_bpm
                except Exception:
                    pass

        frame_placeholder.image(frame, channels="BGR")

        if st.session_state.bpm_locked:
            bpm_val = st.session_state.locked_bpm
            bpm_placeholder.metric("Heart Rate", f"{bpm_val:.1f} BPM", "Locked ✅")
            status_text, status_color = get_status(bpm_val)
            status_placeholder.markdown(f"**Status:** :{status_color}[{status_text}]")
        elif st.session_state.bpm_history:
            bpm_val = np.mean(st.session_state.bpm_history)
            bpm_placeholder.metric("Heart Rate", f"{bpm_val:.1f} BPM", "Calculating...")
            status_text, status_color = get_status(bpm_val)
            status_placeholder.markdown(f"**Status:** :{status_color}[{status_text}]")
        else:
            bpm_placeholder.metric("Heart Rate", "-- BPM", "Calculating...")

        if filtered_for_plot is not None:
            chart_placeholder.line_chart(filtered_for_plot)

    cap.release()
else:
    st.info("Tick 'Start Camera' above to begin. Sit still and face the camera in good lighting for best results.") 