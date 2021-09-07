from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import numpy as np

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

classes = ["COVID", "Healthy", "Pneumonia"]

@app.route("/")
def landing_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def covid_prediction():
    try:
        print("url loaded ..................................")
        if not "image" in request.files:
            print(request.files)
            return "Image not found"
        # Read image
        image = request.files["image"]
        image = Image.open(image)
        image = image.convert("RGB")
        img = image.resize((224, 224))
        
        # Convert to numpy array
        img = img_to_array(img)

        # Expand dims
        img = np.expand_dims(img, axis=0)
        
        datagen = ImageDataGenerator(
            rescale=1.0/255.0, 
            samplewise_center=True,
            samplewise_std_normalization=True)
        
        it = datagen.flow(img, batch_size=1)
        
        model = load_model("./model/model.h5")
        print("============== Model Prediction =================")
        prediction = model.predict(it)
        # prediciton format --> [[covid, healthy, pneumonia]]
        class_index = np.argmax(prediction, axis=1)[0]
        
        print(f"Model prediction: \n score {prediction[0][class_index]} \n class {classes[class_index]}")
        print(f"Prediction {prediction}")
        
        return render_template("index.html", diagonised_class=classes[class_index], score=prediction[0][class_index] )
    except Exception as e:
        print(f"Error: {e}")

if __name__  == "__main__":
    app.run(debug=True)