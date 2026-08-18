"""
Website Deteksi Merokok via Kamera HP (browser-based)
-------------------------------------------------------
Flask + Roboflow Inference API.
Kamera diakses langsung dari BROWSER (HP/laptop) via JavaScript getUserMedia,
karena server tidak selalu punya kamera fisik sendiri.
Browser mengirim frame (base64 JPEG) ke server tiap beberapa ratus ms,
server meneruskan ke Roboflow API untuk deteksi, lalu mengirim balik hasil box.

WAJIB:
  - Isi ROBOFLOW_API_KEY di bawah dengan API key asli dari akun Roboflow kamu
  - Install: pip install flask inference-sdk pillow

Jalankan:
    python app.py
Buka di HP (WiFi sama dengan laptop): http://<IP-LOKAL-LAPTOP>:5000
"""

import base64
import io
import time
from flask import Flask, render_template, request, jsonify
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# ============ KONFIGURASI ============
ROBOFLOW_API_KEY = "SO53jP9tSJsDJCd4l76k"   # API key kamu
ROBOFLOW_MODEL_ID = "cigarette-ghnlk/4"
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
CONF_THRESHOLD = 0.4
# ======================================

if ROBOFLOW_API_KEY == "ISI_API_KEY_KAMU_DI_SINI":
    raise SystemExit(
        "[ERROR] ROBOFLOW_API_KEY belum diisi.\n"
        "        Buka Roboflow -> Settings -> API Keys, copy Private API Key,\n"
        "        lalu isi ke variabel ROBOFLOW_API_KEY di app.py."
    )

client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)

detection_state = {
    "smoking_detected": False,
    "confidence": 0.0,
    "last_update": time.time(),
}


@app.route("/")
def index():
    return render_template("index.html", mode="Roboflow API (kamera browser)")


@app.route("/detect", methods=["POST"])
def detect():
    """Terima 1 frame (base64 JPEG) dari browser, deteksi, kembalikan hasil box."""
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")

    if not image_b64:
        return jsonify({"error": "no image"}), 400

    # Buang prefix "data:image/jpeg;base64," kalau ada
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "invalid base64"}), 400

    tmp_path = "_frame_tmp.jpg"
    with open(tmp_path, "wb") as f:
        f.write(image_bytes)

    try:
        result = client.infer(tmp_path, model_id=ROBOFLOW_MODEL_ID)
    except Exception as e:
        print("[WARNING] Gagal memanggil Roboflow API:", e)
        return jsonify({"error": str(e)}), 500

    boxes = []
    smoking = False
    confidence = 0.0

    for pred in result.get("predictions", []):
        conf = pred.get("confidence", 0.0)
        if conf < CONF_THRESHOLD:
            continue
        cls_name = pred.get("class", "")
        cx, cy = pred.get("x", 0), pred.get("y", 0)
        w, h = pred.get("width", 0), pred.get("height", 0)
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2

        boxes.append({
            "class": cls_name,
            "conf": round(conf, 2),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

        is_cigarette = "cigar" in cls_name.lower() or "smok" in cls_name.lower()
        if is_cigarette:
            smoking = True
            confidence = max(confidence, conf)

    detection_state["smoking_detected"] = smoking
    detection_state["confidence"] = round(confidence, 2)
    detection_state["last_update"] = time.time()

    return jsonify({
        "boxes": boxes,
        "smoking_detected": smoking,
        "confidence": round(confidence, 2),
        "image_width": result.get("image", {}).get("width"),
        "image_height": result.get("image", {}).get("height"),
    })


@app.route("/status")
def status():
    return jsonify(detection_state)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)