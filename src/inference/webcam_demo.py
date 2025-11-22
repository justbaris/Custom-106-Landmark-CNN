import cv2
import mediapipe as mp
import torch
import numpy as np
import time
import os
import onnxruntime as ort
from math import exp

from src.model.custom_cnn import Custom106Net
from src.utils.heatmap import get_max_preds


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def load_onnx_model(path="experiments/checkpoints/model_106.onnx"):
    if not os.path.exists(path):
        print("[ERROR] ONNX file not found:", path)
        return None
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("[INFO] Loaded ONNX model:", path)
    return sess


def load_model():
    model = Custom106Net(num_landmarks=106, channels=32).to(DEVICE)
    ckpt_path = "experiments/checkpoints/best.pth"

    if not os.path.exists(ckpt_path):
        print("[WARN] best.pth not found, loading last.pth")
        ckpt_path = "experiments/checkpoints/last.pth"

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}")
    return model


def export_to_onnx():
    model = Custom106Net(num_landmarks=106, channels=32).to(DEVICE)
    ckpt = "experiments/checkpoints/best.pth"
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    dummy = torch.randn(1, 3, 256, 256).to(DEVICE)
    out_path = "experiments/checkpoints/model_106.onnx"

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["heatmap"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "heatmap": {0: "batch"}}
    )
    print("[INFO] Exported ONNX model to:", out_path)


def preprocess(frame, bbox, input_size=256):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, (input_size, input_size))
    img = crop[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE), crop


def draw_landmarks(frame, bbox, coords):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    for (px, py) in coords:
        cx = int(x1 + px * w)
        cy = int(y1 + py * h)
        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

    return frame


def run_webcam():
    model = load_model()
    onnx_sess = load_onnx_model()
    filters = [OneEuroFilter() for _ in range(106)]

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face detection
        results = mp_face.process(rgb)

        if results.detections:
            for det in results.detections:
                rel = det.location_data.relative_bounding_box

                x1 = int(rel.xmin * w)
                y1 = int(rel.ymin * h)
                bw = int(rel.width * w)
                bh = int(rel.height * h)
                x2 = x1 + bw
                y2 = y1 + bh

                # Clamp values
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                # Preprocess crop
                img_tensor, crop = preprocess(frame, (x1, y1, x2, y2))

                # Run ONNX if available
                if onnx_sess is not None:
                    inp = img_tensor.cpu().numpy()
                    heatmaps = onnx_sess.run(None, {"input": inp})[0]
                else:
                    with torch.no_grad():
                        heatmaps = model(img_tensor).cpu().numpy()

                coords, _ = get_max_preds(heatmaps)
                coords = coords[0]

                # Apply One Euro Filter smoothing
                t = time.time()
                coords = np.array([filters[i].filter(coords[i], t) for i in range(106)])

                # coords: 0–1 normalized landmark positions
                frame = draw_landmarks(frame, (x1, y1, x2, y2), coords)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)

        # FPS
        curr = time.time()
        fps = 1.0 / (curr - prev_time)
        prev_time = curr
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("106 Landmark Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()