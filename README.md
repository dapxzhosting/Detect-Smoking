"""
Website Deteksi Merokok via Kamera
-----------------------------------
Flask + OpenCV + YOLO. Streaming video dari webcam dan menjalankan deteksi
objek rokok/merokok memakai model YOLO custom.

WAJIB:
  - Siapkan file model hasil training di: models/best.pt
    (mis. dilatih di Roboflow dataset "cigarette/smoking detection", atau model sendiri)
  - Install: pip install ultralytics

Jalankan:
    pip install -r requirements.txt
    python app.py
Buka browser: http://localhost:5000
"""

import os
import cv2
import time
import numpy as np
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# ============ KONFIGURASI ============
USE_YOLO = True  # mode YOLO aktif — pastikan models/best.pt tersedia
YOLO_MODEL_PATH = "models/best.pt"
CAMERA_INDEX = 0  # ganti jika kamera lebih dari satu (0,1,2,...)
YOLO_CONF_THRESHOLD = 0.4  # ambang kepercayaan minimum deteksi
# ======================================

# State global untuk status deteksi (dibaca frontend via /status)
detection_state = {
    "smoking_detected": False,
    "confidence": 0.0,
    "last_update": time.time(),
}

# Haar cascade wajah dipakai hanya untuk anotasi visual (opsional), bukan untuk deteksi rokok
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

yolo_model = None
if USE_YOLO:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "[ERROR] Paket 'ultralytics' belum terinstall.\n"
            "        Jalankan: pip install ultralytics"
        )

    if not os.path.exists(YOLO_MODEL_PATH):
        raise SystemExit(
            f"[ERROR] Model tidak ditemukan di '{YOLO_MODEL_PATH}'.\n"
            "        Taruh file bobot (best.pt) hasil training/download ke folder models/,\n"
            "        atau set USE_YOLO = False untuk pakai mode heuristic."
        )

    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("[INFO] Model YOLO berhasil dimuat:", YOLO_MODEL_PATH)


def detect_yolo(frame):
    """Deteksi menggunakan model YOLO custom."""
    results = yolo_model.predict(frame, verbose=False, conf=YOLO_CONF_THRESHOLD)
    smoking = False
    confidence = 0.0

    for r in results:
        for box in r.boxes:
            cls_name = yolo_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if "smok" in cls_name.lower() or "cigar" in cls_name.lower():
                smoking = True
                confidence = max(confidence, conf)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
            cv2.putText(
                frame, f"{cls_name} {conf:.2f}", (xyxy[0], xyxy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

    return frame, smoking, confidence


def gen_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Tidak bisa membuka kamera. Cek CAMERA_INDEX atau izin kamera.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        frame, smoking, confidence = detect_yolo(frame)

        detection_state["smoking_detected"] = smoking
        detection_state["confidence"] = round(confidence, 2)
        detection_state["last_update"] = time.time()

        # Overlay status di video
        label = "MEROKOK TERDETEKSI" if smoking else "Normal"
        color = (0, 0, 255) if smoking else (0, 200, 0)
        cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


@app.route("/")
def index():
    return render_template("index.html", mode="YOLO")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(detection_state)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)