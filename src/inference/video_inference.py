import cv2
import mediapipe as mp
import numpy as np
import time
import torch
import onnxruntime as ort
import os
from tqdm import tqdm
import argparse
from collections import deque

from src.model.custom_cnn import Custom106Net
from src.utils.heatmap import get_max_preds


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === One Euro Filter ===
class OneEuroFilter:
    def __init__(self, freq=60, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.last_time = None
        self.x_prev = None
        self.dx_prev = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * 3.1415 * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t):
        if self.last_time is None:
            self.last_time = t
            self.x_prev = x
            self.dx_prev = 0 * x
            return x

        te = t - self.last_time
        self.last_time = t
        self.freq = 1.0 / te

        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


# === IoU Helper ===
def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(boxA_area + boxB_area - inter)

# === Kalman Filter Track ===
class Track:
    def __init__(self, track_id, bbox, fps, smooth=True):
        self.id = track_id
        self.bbox = bbox
        self.time_since_update = 0
        self.smooth = smooth
        self.filters = [OneEuroFilter(freq=fps) for _ in range(106)] if smooth else None

    def update(self, bbox):
        self.bbox = bbox
        self.time_since_update = 0

    def smooth_landmarks(self, coords, t):
        if not self.smooth:
            return coords
        coords = np.array([self.filters[i].filter(coords[i], t) for i in range(106)])
        return coords


# === ONNX MODEL ===
def load_onnx():
    path = "experiments/checkpoints/model_106.onnx"
    if not os.path.exists(path):
        raise FileNotFoundError("ONNX model not found. Export it first.")

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("[INFO] Loaded ONNX model")
    return sess


def preprocess(frame, bbox, size=256):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (size, size))
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))[None, :]
    return rgb, crop


def draw_landmarks(frame, bbox, coords):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    for (x, y) in coords:
        px = int(x1 + x * w)
        py = int(y1 + y * h)
        cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

    return frame


def run_video(input_path, output_path="output_landmarks.mp4", model_path="experiments/checkpoints/model_106.onnx", show=False, smooth=True):
    print("[INFO] Loading ONNX model...")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    print("[INFO] Loading face detector...")
    mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # smoothing filters for 106 points
    filters = [OneEuroFilter(freq=fps) for _ in range(106)] if smooth else None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    next_track_id = 0
    tracks = []

    print("[INFO] Processing video...")
    pbar = tqdm(total=total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_face.process(rgb)

        detections = []
        if result.detections:
            for det in result.detections:
                rel = det.location_data.relative_bounding_box

                x1 = int(rel.xmin * width)
                y1 = int(rel.ymin * height)
                bw = int(rel.width * width)
                bh = int(rel.height * height)
                x2 = x1 + bw
                y2 = y1 + bh

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(width, x2); y2 = min(height, y2)

                detections.append([x1, y1, x2, y2])

        # === MATCH DETECTIONS TO TRACKS USING IoU ===
        used_dets = set()
        for track in tracks:
            best_iou = 0
            best_det = None
            best_idx = -1

            for i, det in enumerate(detections):
                if i in used_dets:
                    continue
                iou = bbox_iou(track.bbox, det)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
                    best_idx = i

            if best_iou > 0.3:
                track.update(best_det)
                used_dets.add(best_idx)
            else:
                track.time_since_update += 1

        # === CREATE NEW TRACKS FOR UNMATCHED DETECTIONS ===
        for i, det in enumerate(detections):
            if i not in used_dets:
                tracks.append(Track(next_track_id, det, fps=fps, smooth=smooth))
                next_track_id += 1

        # === REMOVE OLD TRACKS ===
        tracks = [t for t in tracks if t.time_since_update < 20]

        # === PROCESS EACH TRACK ===
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            inp, _ = preprocess(frame, track.bbox)

            heatmaps = sess.run(None, {"input": inp})[0]
            coords, _ = get_max_preds(heatmaps)
            coords = coords[0]

            # smoothing per track
            tstamp = time.time()
            coords = track.smooth_landmarks(coords, tstamp)

            frame = draw_landmarks(frame, track.bbox, coords)
            cv2.putText(frame, f"ID {track.id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if show:
            cv2.imshow("Video Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    print(f"[INFO] Video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full video inference for 106 landmarks")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", "-o", type=str, default="output_landmarks.mp4", help="Path to output video")
    parser.add_argument("--model", type=str, default="experiments/checkpoints/model_106.onnx", help="Path to ONNX model")
    parser.add_argument("--show", action="store_true", help="Show live video during processing")
    parser.add_argument("--no_smooth", action="store_true", help="Disable OneEuroFilter smoothing")

    args = parser.parse_args()

    run_video(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        show=args.show,
        smooth=not args.no_smooth
    )