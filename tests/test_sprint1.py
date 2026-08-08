import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Ana dizini path'e ekle ki qillm modülünü bulabilsin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qillm.wrapper import compress_model

MODEL_ID = "Qwen/Qwen2.5-0.5B"

print("1. HuggingFace Modeli Yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

print("\n2. QILLM Sarmalayıcı Çalıştırılıyor...")
compressed_model = compress_model(model, compression_ratio=0.5)

print("\n3. Başarıyla tamamlandı. Sıkıştırılmış model hazır!")