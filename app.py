import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
os.makedirs("templates", exist_ok=True)

with open("templates/index.html", "w") as f:
    f.write("""[PASTE THE index.html ABOVE]""")

with open("templates/result.html", "w") as f:
    f.write("""[PASTE THE result.html ABOVE]""")

# =======================
# Load model
# =======================
MODEL_PATH = "my_model.h5"
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# =======================
# Load class labels from encoded.csv
# =======================
class_dict = {}
try:
    df = pd.read_csv("encoded.csv")
    df = df[['Prediction Labels', 'Prediction Classes']].drop_duplicates()
    class_dict = dict(zip(df['Prediction Labels'], df['Prediction Classes']))
    print("✅ Class dictionary loaded from encoded.csv")
except Exception as e:
    print(f"❌ Error loading encoded.csv: {e}")
    class_dict = {i: f"class_{i}" for i in range(90)}  # fallback

# =======================
# Create basic HTML upload form
# =======================
if not os.path.exists('templates'):
    os.makedirs('templates')

if not os.path.exists('static'):
    os.makedirs('static')

with open('templates/index.html', 'w') as f:
    f.write("""
<!DOCTYPE html>
<html>
<head><title>Animal Classifier</title></head>
<body>
  <h1>Upload an Image for Classification</h1>
  <form method="POST" enctype="multipart/form-data">
    <input type="file" name="file"><br><br>
    <input type="submit" value="Classify">
  </form>
</body>
</html>
""")

# =======================
# Routes
# =======================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file:
            try:
                img_path = os.path.join("static", file.filename)
                file.save(img_path)

                img = Image.open(img_path).resize((224, 224))
                img_array = np.expand_dims(np.array(img), axis=0)
                img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

                prediction = model.predict(img_array)
                predicted_index = int(np.argmax(prediction))
                predicted_class = class_dict.get(predicted_index, f"Unknown_{predicted_index}")
                confidence = float(np.max(prediction))

                return jsonify({
                    "image_url": "/" + img_path,
                    "prediction": predicted_class,
                    "confidence": round(confidence, 4)
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    return render_template('index.html')


# =======================
# Run the app
# =======================
if __name__ == '__main__':
    app.run(debug=True)
