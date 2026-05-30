from fastapi import FastAPI, UploadFile, File

from inference.predict import predict_image

import tempfile
import shutil

app = FastAPI()


@app.get("/")
def home():
    return {
        "status": "running",
        "project": "Math To LaTeX"
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as temp_file:

        shutil.copyfileobj(
            image.file,
            temp_file
        )

        temp_path = temp_file.name

    latex = predict_image(
        temp_path
    )

    return {
        "latex": latex
    }