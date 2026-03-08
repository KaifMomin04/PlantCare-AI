import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

classes = [
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

def predict_disease(img_path):

    img = cv2.imread(img_path)

    # Convert BGR → RGB (important)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224,224))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)

    return classes[class_index]


# Example test
result = predict_disease("src/test_leaf.jpg")
print("Predicted Disease:", result)