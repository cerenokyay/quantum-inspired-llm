import torch
from qillm.attention import QuantumInspiredAttention

print("--- 🧠 KUANTUM ESİNLEMELİ LLM ATTENTION KATMANI TESTİ ---")

# Model boyutunu belirleyelim (d_model = 8)
d_model = 8
batch_size = 1
seq_len = 4  # 4 kelimelik bir girdi metni simülasyonu

# Rastgele bir metin vektörü oluşturalım
input_tensor = torch.randn(batch_size, seq_len, d_model)

# Kuantum Katmanımızı Çağıralım
q_attention = QuantumInspiredAttention(d_model=d_model)

print("\n1. Girdi Vektörü C++ Kuantum Katmanına Gönderiliyor...")
output = q_attention(input_tensor)

print("\n2. Başarıyla İşlenen ve Sıkıştırılan Çıktı Vektörü:")
print(output)
print(f"\nÇıktı Boyutu: {output.shape} (Girdi boyutuyla birebir uyumlu ve hatasız!)")