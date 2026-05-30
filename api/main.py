from fastapi import FastAPI, UploadFile, File

from inference.predict import predict_image

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

import tempfile
import shutil

app = FastAPI()

templates = Jinja2Templates(
    directory = "templates"
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


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