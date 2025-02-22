# 📌 API Documentation

## `Anomaly_detection_py.py`

- **Detects motion-based anomalies using background subtraction and optical flow.**
- **Processes each frame to highlight moving objects.**

### **Functions**

- `load_video(video_path)`: Loads the input video.
- `process_video(input_path, output_path)`: Runs the anomaly detection pipeline.

---

## `detect_anomalies.py`

- **Identifies scratches and stains using edge detection and adaptive thresholding.**
- **Processes each frame and marks detected defects with bounding boxes.**

### **Functions**

- `detect_anomalies(frame)`: Detects scratches and stains.
- `process_video(input_path, output_path)`: Processes the video frame by frame.
