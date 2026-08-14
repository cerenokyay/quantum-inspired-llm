# app.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
import time

app = FastAPI(title="QILLM Engine API", version="2.2.0")

class CompressRequest(BaseModel):
    ratio: float
    model_name: str

class RecoverRequest(BaseModel):
    model_name: str
    method: str

class InferenceRequest(BaseModel):
    prompt: str
    model_state: str 

@app.post("/analyze")
def analyze_model(req: CompressRequest):
    is_small_model = "0.5B" in req.model_name or "1B" in req.model_name or ".pt" in req.model_name
    return {
        "model": req.model_name,
        "is_small": is_small_model,
        "warning_message": f"Dikkat: {req.model_name} gibi küçük/özel modellerde yüksek SVD budaması geçici PPL artışına yol açabilir."
    }

@app.post("/compress")
def compress_endpoint(req: CompressRequest):
    time.sleep(2) 
    orig_vram = 1884.59 if "Qwen" in req.model_name else 4096.0
    comp_vram = orig_vram * req.ratio if req.ratio > 0.3 else orig_vram * 0.45
    
    return {
        "status": "success",
        "ratio_used": req.ratio,
        "original_vram_mb": round(orig_vram, 2),
        "compressed_vram_mb": round(comp_vram, 2),
        "vram_saved_percent": round(((orig_vram - comp_vram) / orig_vram) * 100, 2),
        "original_ppl": 7.21,
        "compressed_ppl": 2866985.12 if req.ratio < 0.5 else 150.4
    }

@app.post("/recover")
def recover_endpoint(req: RecoverRequest):
    time.sleep(3) 
    if req.method == "LoRA (Hızlı İnce Ayar)":
        rec_vram, rec_ppl = 985.0, 9.45
    elif req.method == "QLoRA (Kuantize Düşük Bellek)":
        rec_vram, rec_ppl = 960.0, 12.10
    else: 
        rec_vram, rec_ppl = 953.46, 8.90

    # method_used anahtarının kesinlikle döndüğünden emin oluyoruz
    return {
        "status": "success",
        "method_used": req.method,
        "recovered_vram_mb": rec_vram,
        "recovered_ppl": rec_ppl
    }

@app.post("/inference")
def live_inference(req: InferenceRequest):
    time.sleep(1)
    if req.model_state == "original":
        speed = 20.92
        output = f"{req.prompt} alanında geleneksel yöntemler genellikle yüksek işlem gücü gerektirir. Bu durum model boyutlarıyla doğrudan ilişkilidir."
    elif req.model_state == "compressed":
        speed = 28.11
        output = f"{req.prompt} asdhaks jhasd qweqwe (Sıkıştırma sebebiyle model zekası bozuldu, mantıksız çıktılar üretiyor)."
    else:
        speed = 26.50
        output = f"{req.prompt} alanında kuantum esinlemeli algoritmalar, VRAM kullanımını optimize ederek devasa kazanımlar sağlar ve performansı korur."

    # output anahtarını kesinlikle dönüyoruz
    return {"output": output, "speed": speed}

@app.post("/upload")
def upload_local_model(file: UploadFile = File(...)):
    """Kullanıcının yüklediği yerel modeli kabul eden uç nokta"""
    time.sleep(1) # Yükleme simülasyonu
    return {"status": "success", "filename": file.filename}

@app.get("/download")
def download_model():
    fake_weights = b"QILLM_TENSOR_WEIGHTS_MOCK_DATA_0x8F..."
    return Response(content=fake_weights, media_type="application/octet-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)