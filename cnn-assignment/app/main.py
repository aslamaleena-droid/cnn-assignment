from fastapi import FastAPI, UploadFile, File
from app.infer import load_model, predict_bytes

app = FastAPI(title="CNN Classifier", version="1.0")
model = load_model("model.pth")

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    img = await file.read()
    return predict_bytes(model, img)

@app.get("/sample")
def sample():
    return {"usage":"POST an image file to /classify"}
