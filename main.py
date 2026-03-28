from flask import Flask, render_template, request, jsonify
import os
import requests
import base64

API_KEY = "mgMkXwGrzsH9yjyQMpkm"
MODEL_URL = "https://detect.roboflow.com/ppe-detection-qlq3d/1"

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("camera.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_data = data['image'].split(',')[1]

    response = requests.post(
        MODEL_URL,
        params={"api_key": API_KEY},
        data=base64.b64decode(image_data),
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    result = response.json()

    alerts = []

    for obj in result.get('predictions', []):
        label = obj['class']

        if label == "no helmet":
            alerts.append("⚠️ No Helmet")

        if label == "no vest":
            alerts.append("⚠️ No Vest")

    if not alerts:
        return jsonify({"result": "✅ All Safe"})

    return jsonify({"result": "\n".join(alerts)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
