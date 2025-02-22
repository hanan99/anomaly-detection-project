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
        frame_count (int): Total number of frames.
    """
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_width, frame_height, fps, frame_count

def process_video(input_path, output_path):
    """
    Processes the video to detect motion anomalies using background subtraction
    and optical flow.

    Args:
        input_path (str): Path to the input video.
        output_path (str): Path to save the processed video.
    """
    cap = load_video(input_path)
    frame_width, frame_height, fps, frame_count = get_video_properties(cap)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    prev_gray = None
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(frame)
        _, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
        
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            mask = np.zeros_like(frame)
            mask[..., 1] = 255
            mask[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
            mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            flow_bgr = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            overlay = cv2.addWeighted(frame, 0.7, flow_bgr, 0.3, 0)
        else:
            overlay = frame
        
        prev_gray = gray
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        out.write(overlay)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx} frames out of {frame_count}")
    
    cap.release()
    out.release()
    print(f"[INFO] Processing complete. Video saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection in Video using OpenCV")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the processed video")
    
    args = parser.parse_args()
    process_video(args.input, args.output)
