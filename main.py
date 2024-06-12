from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io

app = FastAPI()

# Statik dosyaları sunma
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Modeli yükleme
model = load_model('model/animal_classifier.h5')

# Sınıf isimleri sözlüğü
class_names = [
    {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider"}
]

# Reverse dictionary for prediction to class name
reverse_class_names = {v: k for d in class_names for k, v in d.items()}

# Güven eşiği
CONFIDENCE_THRESHOLD = 0.8

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("../frontend/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = image.load_img(io.BytesIO(contents), target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Güven eşiği kontrolü
    if confidence < CONFIDENCE_THRESHOLD:
        return {"Result": "Unrecognized"}

    # Get the predicted class name from reverse dictionary
    predicted_class = list(reverse_class_names.keys())[predicted_class_index]
    result = predicted_class

    return {"result": result, "confidence": float(confidence)}
