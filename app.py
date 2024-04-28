# from flask import Flask, request, jsonify
# from werkzeug.utils import secure_filename
# import os
# import predict

# app = Flask(__name__)

# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#         disease = predict.prediction(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

#         return jsonify({'disease': disease})

#     return jsonify({'error': 'Invalid file'})


# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, request, jsonify
# import predict

# app = Flask(__name__)


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     image_data = request.files['photo'].read()
#     predicted_disease = predict.prediction(image_data)
#     return jsonify({'disease': predicted_disease})


# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from PIL import Image
import numpy as np
from io import BytesIO
from flask import Flask, render_template
from flask import Flask, request, jsonify
import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg19 import preprocess_input

app = Flask(__name__)

# Serve static files from the 'public' directory
# app.add_url_rule('/public/<path:path>', 'static', build_only=True)
# app.add_url_rule('/view/css/global.css', 'css_global', build_only=True)
# app.add_url_rule('/view/css/main-desktop.css','css_main_desktop', build_only=True)

# Load your machine learning model here
model_path = os.path.abspath('dermenaop_model.h5')
model = tf.keras.models.load_model(model_path)

# Define disease labels
disease_labels = [
    "Actinic Keratosis", "Atopic Dermatitis", "Benign Keratosis",
    "Dermatofibroma", "Melanoma", "Melanocytic Nevus",
    "Squamous Cell Carcinoma", "Tinea Ringworm Candidiasis",
    "Vascular Lesions"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/login-desktop.html")
def login():
    return render_template("login-desktop.html")


@app.route("/ai_page.html")
def ai_page():
    return render_template("ai_page.html")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Convert the file to a BytesIO object
            img_bytes = file.read()
            img = Image.open(BytesIO(img_bytes))
            # Preprocess the image
            img = img.resize((256, 256))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            # Make the prediction
            pred = model.predict(img_array)
            pred_class = np.argmax(pred)
            predicted_disease = disease_labels[pred_class]
            # Return the prediction as a JSON response
            return jsonify({'disease': predicted_disease})
        except Exception as e:
            return jsonify({'error': str(e)}), 400


@app.route('/result', methods=['GET'])
def result():
    disease = request.args.get('disease')
    return render_template('result.html', disease=disease)


@app.route("/info.html")
def info():
    return render_template("info.html")


@app.route("/ActinicKeratosis.html")
def ActinicKeratosis():
    return render_template("ActinicKeratosis.html")


@app.route("/AtopicDermatitis.html")
def AtopicDermatitis():
    return render_template("AtopicDermatitis.html")


@app.route("/BenignKeratosis.html")
def BenignKeratosis():
    return render_template("BenignKeratosis.html")


@app.route("/Dermatofibroma.html")
def Dermatofibroma():
    return render_template("Dermatofibroma.html")


@app.route("/MelanocyticNevus.html")
def MelanocyticNevus():
    return render_template("MelanocyticNevus.html")


@app.route("/Melanomaa.html")
def Melanomaa():
    return render_template("Melanomaa.html")


@app.route("/SquamousCellCarcinoma.html")
def SquamousCellCarcinoma():
    return render_template("SquamousCellCarcinoma.html")


@app.route("/TineaRingwormCandidiasis.html")
def TineaRingwormCandidiasis():
    return render_template("TineaRingwormCandidiasis.html")


@app.route("/Vascularlesion.html")
def Vascularlesion():
    return render_template("Vascularlesion.html")


@app.route("/about.html")
def about():
    return render_template("about.html")


@app.route("/skin-health.html")
def  skin_health():
    return render_template("/skin-health.html")


@app.route("/upload-photo-8.html")
def Clinical_Solution():
    return render_template("/upload-photo-8.html")

if __name__ == '__main__':
    app.run(debug=True)






