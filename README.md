# Contactless Heart Rate Monitor

Estimate heart rate from a face using only a laptop webcam. No wearable and no sensor needed.

> Wellness-screening prototype. Not a medical device.

## How it works

Each heartbeat changes blood volume under the skin, which causes a tiny periodic change in skin colour. The change is invisible to the eye but measurable in video (remote photoplethysmography, or rPPG).

1. **Face detection**: MediaPipe finds the face in each webcam frame
2. **Region of interest**: a forehead region is tracked
3. **Signal extraction**: the average green-channel intensity per frame forms a time series
4. **Filtering**: a band-pass filter (0.7-3 Hz) keeps the plausible heart-rate range
5. **FFT**: the dominant frequency is converted to beats per minute
6. **Stabilisation**: recent estimates are averaged, and the reading locks once they agree

## Features

- Live webcam feed with real-time BPM
- Pulse waveform graph
- Status label (Low / Normal / Elevated)
- Auto-lock on a stable reading, with a reset button
- Streamlit web interface

## Tech stack

Python 3.11, OpenCV, MediaPipe, NumPy, SciPy, Matplotlib, Streamlit

## Run it locally

```bash
git clone https://github.com/YOUR-USERNAME/heart-rate-monitor.git
cd heart-rate-monitor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Tick **Start Camera**, sit still in steady lighting, and wait 10-15 seconds.

## Tips for best results

- Face a window or lamp, and avoid backlighting
- Stay still while measuring
- Avoid flickering lights

## Limitations

- Sensitive to lighting and head movement
- Tested on a limited number of people
- Not a medical device

## Future work

- Test across more people and lighting conditions
- Compare readings against a reference device
- Add a signal-quality estimate
- Improve robustness to motion

## Acknowledgements

Built on the established rPPG technique and the open-source Python, OpenCV, MediaPipe and Streamlit libraries.

## Author

**CR Devika**, BSc Computer Science (Hons) in AI & Robotics, Chinmaya Vishwa Vidyapeeth university, Kochi
