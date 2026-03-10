import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from collections import deque
import argparse
import sys

# Lightweight Inference Engines
import onnxruntime as ort
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_inference(video_source, lstm_onnx_path='../models/accident_detection_lstm_model.onnx', yolo_weights='yolov8n.pt', output=None):
    if not os.path.exists(lstm_onnx_path):
        print(f"Error: ONNX model not found at {lstm_onnx_path}. Use convert_lstm.py first.")
        sys.exit(1)
        
    print(f"[INFO] Loading YOLO weights from {yolo_weights}...")
    detector = YOLO(yolo_weights)
    
    print("[INFO] Initializing Deep SORT Tracker...")
    tracker = DeepSort(max_age=30, nn_budget=100)
    
    print(f"[INFO] Loading ONNX LSTM model from {lstm_onnx_path}...")
    # Initialize ONNX runtime session (CPU only)
    ort_session = ort.InferenceSession(lstm_onnx_path, providers=['CPUExecutionProvider'])
    input_meta = ort_session.get_inputs()[0]
    input_name = input_meta.name
    
    # Dynamically detect sequence length (e.g., 30 or 50)
    seq_length = 50 
    print(f"[INFO] Using fixed sequence length: {seq_length}")
    
    vehicle_classes = [2, 3, 5, 7]
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open {video_source}")
        sys.exit(1)
        
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
    out_writer = None
    if output:
        # Try avc1 first
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
        
        if not out_writer.isOpened():
            print(f"[INFO] avc1 failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
            
        if not out_writer.isOpened():
            print(f"[ERROR] Could not open VideoWriter for {output}")
        else:
            print(f"[INFO] VideoWriter opened successfully for {output}")
    
    recent_features = deque(maxlen=seq_length)
    FRAME_DIM = 114
    last_marker_time = -10.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        vis_frame = frame.copy()
        
        # YOLO Detection
        results = detector(vis_frame, classes=vehicle_classes, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # Strict Boundary Enforcement
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(vis_frame.shape[1], int(x2)), min(vis_frame.shape[0], int(y2))
                w, h = x2 - x1, y2 - y1
                if w > 5 and h > 5:
                    detections.append(([x1, y1, w, h], conf, cls))
        
        # DeepSORT Tracking
        tracks = tracker.update_tracks(detections, frame=vis_frame)
        current_frame_features = []
        
        tracked_vehicles = [t for t in tracks if t.is_confirmed()]
        # Sort by biggest vehicles first
        tracked_vehicles = sorted(tracked_vehicles, key=lambda t: (t.to_ltrb()[2]-t.to_ltrb()[0])*(t.to_ltrb()[3]-t.to_ltrb()[1]), reverse=True)
        
        MAX_VEHICLES = 19
        flattened = []
        
        for i in range(MAX_VEHICLES):
            if i < len(tracked_vehicles):
                track = tracked_vehicles[i]
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                # Feature vector MUST perfectly match training data: [x1, y1, x2, y2, 0.9, 2.0]
                flattened.extend([ltrb[0], ltrb[1], ltrb[2], ltrb[3], 0.9, 2.0])
                
                # Draw on frame
                x1_d, y1_d = int(ltrb[0]), int(ltrb[1])
                x2_d, y2_d = int(ltrb[2]), int(ltrb[3])
                cv2.rectangle(vis_frame, (x1_d, y1_d), (x2_d, y2_d), (0, 255, 0), 2)
                
                # Draw ID label with background
                label = f"ID:{track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_frame, (x1_d, y1_d - th - 8), (x1_d + tw + 4, y1_d), (0, 255, 0), -1)
                cv2.putText(vis_frame, label, (x1_d + 2, y1_d - 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                # Pad to 19 vehicles with zeros
                flattened.extend([0.0] * 6)
                
        assert len(flattened) == FRAME_DIM
        recent_features.append(flattened)
        
        # LSTM Prediction via ONNX
        prob = 0.0
        if len(recent_features) == seq_length:
            input_data = np.array([recent_features], dtype=np.float32)
            outputs = ort_session.run(None, {input_name: input_data})
            prob = float(outputs[0][0][0])
            
        accident_pct = prob * 100
        
        if prob > 0.5:
            banner_text = f"Accident  {accident_pct:.2f}"
            text_color = (0, 255, 255)        # Cyan text
            bg_color = (0, 0, 255)             # Red background
            # Draw thick red border around the whole frame
            cv2.rectangle(vis_frame, (0, 0), (vis_frame.shape[1]-1, vis_frame.shape[0]-1), (0, 0, 255), 4)
            
            # Print marker for Node.js to consume (with 3 second cooldown)
            if current_time_sec - last_marker_time > 3.0:
                print(f"MARKER:{current_time_sec:.1f}:{accident_pct:.1f}:vehicle", flush=True)
                last_marker_time = current_time_sec
        else:
            banner_text = f"Normal  {100 - accident_pct:.2f}"
            text_color = (0, 255, 0)           # Green text
            bg_color = (0, 100, 0)             # Dark green background
        
        # Position the label at the centroid of all tracked vehicles
        if len(tracked_vehicles) > 0:
            cx_sum, cy_sum = 0, 0
            for t in tracked_vehicles:
                lb = t.to_ltrb()
                cx_sum += (lb[0] + lb[2]) / 2
                cy_sum += (lb[1] + lb[3]) / 2
            label_x = int(cx_sum / len(tracked_vehicles))
            label_y = int(cy_sum / len(tracked_vehicles))
        else:
            label_x = vis_frame.shape[1] // 2
            label_y = vis_frame.shape[0] // 2
        
        # Draw label with background at vehicle centroid
        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        lx = max(0, label_x - tw // 2)
        ly = max(th + 10, label_y - 20)
        cv2.rectangle(vis_frame, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), bg_color, -1)
        cv2.putText(vis_frame, banner_text, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)
        
        # Show tracked vehicle count
        vehicle_count_text = f"Vehicles Tracked: {len(tracked_vehicles)}"
        cv2.putText(vis_frame, vehicle_count_text, (20, vis_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Base64 for live preview (Throttled to every 2nd frame)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 2 == 0:
            _, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            print(f"FRAME:{base64.b64encode(buffer).decode('utf-8')}")
        
        if out_writer:
            out_writer.write(vis_frame)

    cap.release()
    if out_writer: out_writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    import base64
    run_inference(args.video, output=args.output)
