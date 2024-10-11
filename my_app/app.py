from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('20240930-11431727696637-epoch_50.h5')

# Define image size for the model
IMG_SIZE = 224

# Initialize the FastAPI app
app = FastAPI()

# Function to preprocess the image
def process_image(image) -> np.ndarray:
    image = image.resize((IMG_SIZE, IMG_SIZE))  # Resize the image
    image = np.array(image) / 255.0             # Normalize pixel values
    image = np.expand_dims(image, axis=0)       # Add batch dimension
    return image

# Define a route for prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the uploaded image
        image = Image.open(file.file)
        processed_image = process_image(image)

        # Make a prediction
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions, axis=1)[0]

        # Get the top predictions for visualization
        top_indices = predictions[0].argsort()[-10:][::-1]
        top_predictions = [
            {"breed": f"Breed {i}", "confidence": float(predictions[0][i])}
            for i in top_indices
        ]

        # Mock the correct label for now (for demonstration purposes)
        # In a real scenario, you would compare with a true label if available
        correct_label = "Some Correct Breed"  # Replace this as needed
        is_correct = "Breed " + str(predicted_index) == correct_label

        # Return the results as JSON
        return {
            "predicted_class": f"Breed {predicted_index}",
            "confidence": float(predictions[0][predicted_index]),
            "top_predictions": top_predictions,
            "is_correct": is_correct,
            "correct_label": correct_label
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
