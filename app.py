import os
import pickle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- SETUP SESUAI HALAMAN 2 ---
# path setup (PENTING)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# load model
MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")

# Pastikan file model ada sebelum di-load
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# inisialisasi FastAPI
app = FastAPI(
    title="Spam Email Classification API",
    description="API klasifikasi email spam / ham menggunakan SVM",
    version="1.0.0"
)

# templates
templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)

# --- SKEMA DATA SESUAI HALAMAN 3 ---
class EmailRequest(BaseModel):
    email: str

class EmailResponse(BaseModel):
    prediction: str
    confidence: float

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/predict", response_model=EmailResponse)
async def predict_email(request: EmailRequest):
    # Logika prediksi sesuai Halaman 3
    X = [request.email]
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()
    
    return {
        "prediction": str(prediction),
        "confidence": round(float(probability), 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)