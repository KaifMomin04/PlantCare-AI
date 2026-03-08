from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static"
)

UPLOAD_FOLDER = "app/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

IMG_SIZE = (224, 224)

# Dataset classes
class_names = [
    "Brinjal",
    "Cabbage BlackRots",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Grape Rot",
    "Ladies finger Bacterial Disease",
    "Potato Early blight",
    "Potato healthy",
    "Potato Lateblight",
    "Red Chilli Anthracnose",
    "Soybean healthy"
]

# Home page
@app.route("/")
def index():
    return render_template("index.html")


# About page
@app.route("/about")
def about():
    return render_template("about.html")


# Upload page
@app.route("/upload")
def upload():
    return render_template("upload.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    predicted_class = class_names[class_index]
    confidence = round(np.max(prediction) * 100, 2)

    # Health logic
    if "healthy" in predicted_class.lower():
        health_status = "Healthy"
    else:
        health_status = "Disease Detected"

    diagnosis = f"The AI model detected {predicted_class} with {confidence}% confidence."

    recommendations = [
        "Remove infected leaves",
        "Avoid overwatering",
        "Apply recommended fungicide",
        "Ensure proper sunlight and airflow"
    ]

    return render_template(
        "result.html",
        image_url="/static/uploads/" + file.filename,
        plant_name=predicted_class,
        health_status=health_status,
        diagnosis=diagnosis,
        recommendations=recommendations
    )


# Optional result route
@app.route("/result")
def result():
    return render_template("result.html")


if __name__ == "__main__":
    app.run(debug=True)