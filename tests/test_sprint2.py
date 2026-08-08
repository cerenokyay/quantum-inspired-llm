import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qillm.wrapper import compress_model
from qillm.evaluator import QILLMEvaluator

MODEL_ID = "Qwen/Qwen2.5-0.5B"
print("⏳ Modeller Yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

test_text = "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics."
prompt = "Quantum mechanics and neural networks are"

print("\n📊 --- ORİJİNAL MODEL METRİKLERİ ---")
orig_size = QILLMEvaluator.get_model_size_mb(model)
orig_ppl = QILLMEvaluator.calculate_perplexity(model, tokenizer, test_text)
orig_speed = QILLMEvaluator.measure_speed(model, tokenizer, prompt, 30)

print(f"Bellek Ayak İzi : {orig_size:.2f} MB")
print(f"Perplexity (PPL): {orig_ppl:.2f}")
print(f"Üretim Hızı     : {orig_speed:.2f} token/sn")

print("\n⚙️ QILLM C++ Sıkıştırması Uygulanıyor (%50 Ratio)...")
#compressed_model = compress_model(model, compression_ratio=0.5)
# 28. satırdaki compression_ratio değerini 0.85 (Sadece %15 küçültme) yapalım
compressed_model = compress_model(model, compression_ratio=0.25, target_layers=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])

print("\n🚀 --- QILLM SIKIŞTIRILMIŞ MODEL METRİKLERİ ---")
comp_size = QILLMEvaluator.get_model_size_mb(compressed_model)
comp_ppl = QILLMEvaluator.calculate_perplexity(compressed_model, tokenizer, test_text)
comp_speed = QILLMEvaluator.measure_speed(compressed_model, tokenizer, prompt, 30)

tasarruf = ((orig_size - comp_size) / orig_size) * 100

print(f"Yeni Bellek Ayak İzi : {comp_size:.2f} MB (🏆 TASARRUF: % {tasarruf:.2f})")
print(f"Yeni Perplexity (PPL): {comp_ppl:.2f} (Kalite kaybı minimum olmalı)")
print(f"Yeni Üretim Hızı     : {comp_speed:.2f} token/sn")