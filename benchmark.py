import time
import torch
import torch.nn as nn
import math
from qillm.attention import QuantumInspiredAttention

# Standart PyTorch Attention (Karşılaştırma için)
class StandardPyTorchAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

print("--- ⚡ C++ KUANTUM SVD vs STANDART PYTORCH BENCHMARK ---")

# Test Parametreleri (Büyük Matris Simülasyonu)
batch_size = 8
seq_len = 128  # Uzun cümle/metin simülasyonu
d_model = 64
iterations = 100

dummy_input = torch.randn(batch_size, seq_len, d_model)

# Model Katmanları
standard_attn = StandardPyTorchAttention(d_model)
quantum_cpp_attn = QuantumInspiredAttention(d_model, compression_rate=0.5)

# Warm-up (Isınma Turları)
for _ in range(10):
    _ = standard_attn(dummy_input)
    _ = quantum_cpp_attn(dummy_input)

# 1. STANDART PYTORCH ATTENTION ÖLÇÜMÜ
start_time = time.perf_counter()
for _ in range(iterations):
    _ = standard_attn(dummy_input)
end_time = time.perf_counter()
py_time = (end_time - start_time) * 1000 / iterations

# 2. C++ KUANTUM SVD ATTENTION ÖLÇÜMÜ
start_time = time.perf_counter()
for _ in range(iterations):
    _ = quantum_cpp_attn(dummy_input)
end_time = time.perf_counter()
cpp_time = (end_time - start_time) * 1000 / iterations

print("\n📊 PERFORMANS KARŞILAŞTIRMA SONUÇLARI (100 ITERATION AVG):")
print("-" * 60)
print(f"| Katman Tipi                         | Ortalama Süre (ms) |")
print("-" * 60)
print(f"| Saf PyTorch Attention                | {py_time:8.4f} ms       |")
print(f"| C++ Kuantum SVD Motoru (%50 Sıkıştırma)| {cpp_time:8.4f} ms       |")
print("-" * 60)

diff = abs(py_time - cpp_time)
print(f"\n💡 Özet: C++ Kuantum SVD Katmanı, matrisleri %50 sıkıştırarak tensör büzmesi gerçekleştirdi.")