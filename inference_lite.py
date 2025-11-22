from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import threading
import time
import tensorflow as tf
import joblib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = "models/model_final.tflite"
SCALER_PATH = "scalers/feature_scalers.joblib"

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 2  # gravity + temperature

# Load scaler
scalers = joblib.load(SCALER_PATH)
gravity_scaler = scalers["gravity"]
temp_scaler = scalers["temperature"]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Thread-safe buffer
lock = threading.Lock()
sequence_buffer = []


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Validate input
    if "gravity" not in data or "temperature" not in data:
        return jsonify({"error": "Missing 'gravity' or 'temperature'"}), 400

    gravity = float(data["gravity"])
    temp = float(data["temperature"])

    # Scale input
    gravity_scaled = gravity_scaler.transform([[gravity]])[0][0]
    temp_scaled = temp_scaler.transform([[temp]])[0][0]

    # Push scaled reading to sequence buffer
    with lock:
        sequence_buffer.append([gravity_scaled, temp_scaled])

        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)

        if len(sequence_buffer) < SEQUENCE_LENGTH:
            return jsonify({
                "status": "waiting_for_more_data",
                "received": len(sequence_buffer),
                "required": SEQUENCE_LENGTH
            }), 200

        input_seq = np.array(sequence_buffer, dtype=np.float32)
        input_seq = np.expand_dims(input_seq, axis=0)  # (1, 30, 2)

    # Run inference
    interpreter.set_tensor(input_index, input_seq)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)

    prediction_value = float(prediction.squeeze())

    return jsonify({
        "prediction": prediction_value,
        "timestamp": int(time.time())
    })

@app.route("/reset", methods=["POST"])
def reset_buffer():
    with lock:
        sequence_buffer.clear()
    return jsonify({"status": "buffer_cleared"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000)
