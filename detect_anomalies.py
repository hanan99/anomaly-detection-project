import cv2
import numpy as np
import argparse

def load_video(video_path):
    """
    Loads a video file using OpenCV.

    Args:
        video_path (str): Path to the video file.

    Returns:
        cap (cv2.VideoCapture): OpenCV video capture object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Unable to open video file {video_path}. Please check the path.")
    return cap

def get_video_properties(cap):
    """
    Retrieves video properties such as width, height, and FPS.

    Args:
        cap (cv2.VideoCapture): Video capture object.

    Returns:
        frame_width (int): Width of the video.
        frame_height (int): Height of the video.
        fps (int): Frames per second.
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return frame_width, frame_height, fps

def detect_anomalies(frame):
    """
    Detects scratches and stains in a video frame using edge detection and adaptive thresholding.

    Args:
        frame (numpy.ndarray): The video frame to process.

    Returns:
        numpy.ndarray: The processed frame with detected anomalies marked.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Enhanced scratch detection
    edges = cv2.Canny(blurred, 30, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Scratch", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Stain detection using adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    stain_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in stain_contours:
        if cv2.contourArea(cnt) > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Stain", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def process_video(input_path, output_path):
    """
    Processes the video to detect anomalies and saves the output.
    """
    cap = load_video(input_path)
    frame_width, frame_height, fps = get_video_properties(cap)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        processed_frame = detect_anomalies(frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    print(f"[INFO] Processing complete. Video saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defect Detection in Fabric Video")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the processed video")
    args = parser.parse_args()
    process_video(args.input, args.output)
